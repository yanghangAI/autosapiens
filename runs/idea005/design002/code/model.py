"""Sapiens-based RGBD pose estimation model.

Design 002 — relative_depth_bias:
  Injects a learned depth-based additive bias into the Pose3DHead cross-attention.
  Each joint query's attention over the 960 backbone tokens is modulated by the
  depth value of each corresponding patch via a small Linear(1, num_joints) projection.
  Zero-initialized so the network starts from baseline cross-attention behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Make mmpretrain importable ────────────────────────────────────────────────
_SAPIENS_ROOT = Path("/work/pi_nwycoff_umass_edu/hang/auto/sapiens")
for _p in [str(_SAPIENS_ROOT / "pretrain"), str(_SAPIENS_ROOT / "engine")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpretrain.models.backbones.vision_transformer import VisionTransformer

from infra import NUM_JOINTS, SAPIENS_ARCHS, _interp_pos_embed


# ── BACKBONE ──────────────────────────────────────────────────────────────────

class SapiensBackboneRGBD(nn.Module):
    """Sapiens ViT with a 4-channel (RGB+D) patch embedding."""

    def __init__(self, arch="sapiens_0.3b", img_size=(384, 640), drop_path_rate=0.0):
        super().__init__()
        if arch not in SAPIENS_ARCHS:
            raise ValueError(f"Unknown arch '{arch}'. Choose from {list(SAPIENS_ARCHS)}")
        self.arch      = arch
        self.embed_dim = SAPIENS_ARCHS[arch]["embed_dim"]
        self.vit = VisionTransformer(
            arch=arch, img_size=img_size, patch_size=16, in_channels=4,
            qkv_bias=True, final_norm=True, drop_path_rate=drop_path_rate,
            with_cls_token=False, out_type="featmap", patch_cfg=dict(padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)[0]


# ── DEPTH ATTENTION BIAS MODULE ───────────────────────────────────────────────

class DepthAttentionBias(nn.Module):
    """Computes an additive depth-based bias for cross-attention logits.

    Given a depth map of shape (B, 1, H, W), computes a bias tensor of shape
    (B, num_joints, N_mem) where N_mem = H_tok * W_tok (e.g., 40*24 = 960).

    Parameters:
        depth_proj: Linear(1, num_joints) — zero-initialized

    The bias is head-agnostic (shared across all attention heads).
    """

    def __init__(self, num_joints: int = NUM_JOINTS, patch_size: int = 16):
        super().__init__()
        self.num_joints = num_joints
        self.patch_size = patch_size
        # Linear(1, num_joints): maps each patch depth scalar to per-joint bias
        self.depth_proj = nn.Linear(1, num_joints)
        nn.init.zeros_(self.depth_proj.weight)
        nn.init.zeros_(self.depth_proj.bias)

    def forward(self, depth_ch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_ch: (B, 1, H, W) — normalized depth channel from RGBD input
        Returns:
            bias: (B, num_joints, N_mem) — additive bias for cross-attention logits
        """
        B = depth_ch.size(0)
        # Average-pool to patch grid: (B, 1, H_tok, W_tok)
        depth_patches = F.avg_pool2d(depth_ch, kernel_size=self.patch_size,
                                     stride=self.patch_size)
        # (B, 1, H_tok, W_tok) → (B, N_mem, 1)
        depth_flat = depth_patches.reshape(B, 1, -1).permute(0, 2, 1)  # (B, N_mem, 1)
        # Project: (B, N_mem, 1) → (B, N_mem, num_joints)
        bias = self.depth_proj(depth_flat)          # (B, N_mem, num_joints)
        bias = bias.permute(0, 2, 1)               # (B, num_joints, N_mem)
        return bias


# ── HEAD ──────────────────────────────────────────────────────────────────────

class Pose3DHead(nn.Module):
    """Transformer decoder head with depth-attention bias injection.

    Learnable joint queries (num_joints × hidden_dim) cross-attend to the
    flattened backbone feature map. The cross-attention logits are augmented
    with a learned depth-based additive bias from DepthAttentionBias.

    Architecture:
        input_proj : Linear(in_channels → hidden_dim)
        joint_queries : Embedding(num_joints, hidden_dim)
        depth_attn_bias : DepthAttentionBias
        decoder : TransformerDecoder (num_layers layers, num_heads heads)
        joints_out : Linear(hidden_dim → 3)
    """

    def __init__(self, in_channels, num_joints=NUM_JOINTS, hidden_dim=256,
                 num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.num_joints   = num_joints
        self.num_heads    = num_heads
        self.input_proj   = nn.Linear(in_channels, hidden_dim)
        self.joint_queries = nn.Embedding(num_joints, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder    = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.joints_out = nn.Linear(hidden_dim, 3)
        # Pelvis branches: query[0] (pelvis token) → depth + UV
        self.depth_out  = nn.Linear(hidden_dim, 1)   # (B, 1) forward dist in metres
        self.uv_out     = nn.Linear(hidden_dim, 2)   # (B, 2) normalised [-1, 1]
        # Depth attention bias module (zero-init → starts as baseline)
        self.depth_attn_bias = DepthAttentionBias(num_joints=num_joints)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.joint_queries.weight, std=0.02)
        for m in [self.joints_out, self.depth_out, self.uv_out]:
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor, depth_ch: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            feat: (B, C, H_tok, W_tok) — backbone feature map
            depth_ch: (B, 1, H, W) — normalized depth channel from RGBD input
        Returns:
            dict with 'joints', 'pelvis_depth', 'pelvis_uv'
        """
        B = feat.size(0)
        N_mem = feat.shape[2] * feat.shape[3]  # H_tok * W_tok = 960

        # (B, C, H, W) → (B, H*W, hidden_dim)
        memory = self.input_proj(feat.flatten(2).transpose(1, 2))
        # (B, num_joints, hidden_dim)
        queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)

        # Compute depth bias once: (B, num_joints, N_mem)
        depth_bias = self.depth_attn_bias(depth_ch)  # (B, num_joints, N_mem)

        # Expand for multi-head: (B, num_heads, num_joints, N_mem) → (B*num_heads, num_joints, N_mem)
        depth_bias_expanded = depth_bias.unsqueeze(1).expand(
            B, self.num_heads, -1, -1
        ).reshape(B * self.num_heads, self.num_joints, N_mem)

        # Manual decoder loop with depth bias injection into cross-attention
        tgt = queries  # (B, num_joints, hidden_dim) — batch_first=True

        for layer in self.decoder.layers:
            # 1. Self-attention sublayer (unchanged, with norm_first=True)
            tgt2 = layer.norm1(tgt)
            tgt2, _ = layer.self_attn(tgt2, tgt2, tgt2)
            tgt = tgt + layer.dropout1(tgt2)

            # 2. Cross-attention sublayer with depth bias injection
            tgt2 = layer.norm2(tgt)
            tgt2, _ = layer.multihead_attn(
                tgt2, memory, memory,
                attn_mask=depth_bias_expanded,  # additive bias: (B*nH, num_joints, N_mem)
            )
            tgt = tgt + layer.dropout2(tgt2)

            # 3. FFN sublayer (unchanged, with norm_first=True)
            tgt2 = layer.norm3(tgt)
            tgt2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(tgt2))))
            tgt = tgt + layer.dropout3(tgt2)

        if self.decoder.norm is not None:
            tgt = self.decoder.norm(tgt)

        out = tgt  # (B, num_joints, hidden_dim)
        pelvis_token = out[:, 0, :]                       # (B, hidden_dim) — pelvis query
        return {
            "joints":       self.joints_out(out),         # (B, num_joints, 3)
            "pelvis_depth": self.depth_out(pelvis_token), # (B, 1)
            "pelvis_uv":    self.uv_out(pelvis_token),    # (B, 2)
        }


# ── WEIGHT LOADING ────────────────────────────────────────────────────────────

def load_sapiens_pretrained(model: nn.Module, ckpt_path: str, verbose: bool = True) -> None:
    """Load Sapiens RGB pretrain checkpoint → expand patch embed 3→4ch + interp pos_embed."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    src = raw.get("state_dict", raw.get("model", raw))

    remapped = {f"backbone.vit.{k}": v for k, v in src.items()}
    model_sd = model.state_dict()

    # 1. Expand patch embed 3 → 4 channels
    pe_key = "backbone.vit.patch_embed.projection.weight"
    if pe_key in remapped:
        w_rgb = remapped[pe_key]
        remapped[pe_key] = torch.cat([w_rgb, w_rgb.mean(dim=1, keepdim=True)], dim=1)
        if verbose:
            print("[weights] patch_embed: 3ch → 4ch  (depth = mean of RGB channels)")

    # 2. Interpolate pos_embed
    pe_key = "backbone.vit.pos_embed"
    if pe_key in remapped and pe_key in model_sd:
        src_pe = remapped[pe_key]
        try:
            tgt_h, tgt_w = model.backbone.vit.patch_resolution
        except AttributeError:
            tgt_N = model_sd[pe_key].shape[1]
            tgt_h = tgt_w = int(tgt_N ** 0.5)
        src_g = int((src_pe.shape[1] - 1) ** 0.5)
        remapped[pe_key] = _interp_pos_embed(src_pe, tgt_h, tgt_w, has_cls=True).to(src_pe.dtype)
        if verbose:
            print(f"[weights] pos_embed: {src_g}×{src_g} → {tgt_h}×{tgt_w}  (bicubic)")

    remapped.pop("backbone.vit.cls_token", None)

    load_sd, missing, shape_mismatch = {}, [], []
    for k, v_model in model_sd.items():
        if not k.startswith("backbone."):
            continue
        if k not in remapped:
            missing.append(k); continue
        if v_model.shape != remapped[k].shape:
            shape_mismatch.append(f"  {k}: model {v_model.shape} vs ckpt {remapped[k].shape}"); continue
        load_sd[k] = remapped[k]

    model.load_state_dict(load_sd, strict=False)
    if verbose:
        n_bb = sum(1 for k in model_sd if k.startswith("backbone."))
        print(f"[weights] Loaded {len(load_sd)} / {n_bb} backbone tensors")
        print(f"[weights] Head ({sum(1 for k in model_sd if k.startswith('head.'))} tensors) randomly initialised")
        if missing:
            print(f"[weights] Missing ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if shape_mismatch:
            print(f"[weights] Shape mismatch ({len(shape_mismatch)}):")
            for s in shape_mismatch: print(s)


# ── FULL MODEL ────────────────────────────────────────────────────────────────

class SapiensPose3D(nn.Module):
    """Backbone + Head wrapper. Input (B,4,H,W) → dict of joints/pelvis predictions.

    Design 002: depth channel is extracted and passed to Pose3DHead for
    depth-attention bias injection into cross-attention.
    """

    def __init__(self, arch="sapiens_0.3b", img_size=(640, 384), num_joints=NUM_JOINTS,
                 head_hidden=256, head_num_heads=8, head_num_layers=4,
                 head_dropout=0.1, drop_path_rate=0.0):
        super().__init__()
        self.backbone = SapiensBackboneRGBD(arch=arch, img_size=img_size,
                                             drop_path_rate=drop_path_rate)
        self.head = Pose3DHead(in_channels=SAPIENS_ARCHS[arch]["embed_dim"],
                               num_joints=num_joints, hidden_dim=head_hidden,
                               num_heads=head_num_heads, num_layers=head_num_layers,
                               dropout=head_dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Extract depth channel before passing to backbone
        depth_ch = x[:, 3:4, :, :]  # (B, 1, H, W)
        feat = self.backbone(x)     # (B, C, H_tok, W_tok)
        return self.head(feat, depth_ch)

    def load_pretrained(self, ckpt_path: str, verbose: bool = True) -> None:
        load_sapiens_pretrained(self, ckpt_path, verbose=verbose)
