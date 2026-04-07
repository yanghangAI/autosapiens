"""Sapiens-based RGBD pose estimation model.

Contains the backbone, head, weight loading, and full model wrapper.
Modify this file when changing model architecture or depth-embedding strategy.
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


# ── DEPTH-CONDITIONED POSITIONAL EMBEDDING ────────────────────────────────────

class DepthConditionedPE(nn.Module):
    """3-layer MLP: (B, N, 3) → (B, N, embed_dim).

    Input coordinates: [row_norm ∈ [0,1], col_norm ∈ [0,1], depth_norm ∈ [0,1]]
    Output: additive PE correction to add on top of the standard pos_embed.
    Initialized with very small gain (0.01) so output starts near-zero.
    """

    def __init__(self, embed_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, embed_dim)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight, gain=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (B, N, 3) → output: (B, N, embed_dim)"""
        x = F.gelu(self.fc1(coords))
        x = F.gelu(self.fc2(x))
        return self.fc3(x)


# ── BACKBONE ──────────────────────────────────────────────────────────────────

class SapiensBackboneRGBD(nn.Module):
    """Sapiens ViT with a 4-channel (RGB+D) patch embedding.

    Design 003: depth-conditioned positional embedding (continuous MLP-based).
    The standard pretrained pos_embed is kept as a fixed additive base;
    a 3-layer MLP learns a residual correction conditioned on depth content.
    """

    # Patch grid dimensions matching ViT patch_resolution with padding=2 and stride=16
    H_TOK = 40
    W_TOK = 24

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
        # Depth-conditioned PE MLP (residual correction on top of pretrained pos_embed)
        self.depth_cond_pe = DepthConditionedPE(embed_dim=self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Manual forward: patch_embed → (pe_base + pe_correction) → layers → norm.
        # pos_embed added here; vit.forward() is NOT called to avoid double-adding.
        """
        B = x.shape[0]
        H_tok, W_tok = self.H_TOK, self.W_TOK

        # 1. Compute depth patch averages for the MLP input
        depth_ch = x[:, 3:4, :, :]                                         # (B, 1, 640, 384)
        depth_patches = F.avg_pool2d(depth_ch, kernel_size=16, stride=16)   # (B, 1, 40, 24)
        depth_flat = depth_patches.reshape(B, -1)                           # (B, 960)

        # 2. Build normalized (row, col) coordinate grid
        rows = torch.arange(H_tok, device=x.device, dtype=torch.float32) / (H_tok - 1)  # (40,)
        cols = torch.arange(W_tok, device=x.device, dtype=torch.float32) / (W_tok - 1)  # (24,)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing='ij')                   # (40, 24)
        row_flat = row_grid.reshape(-1)   # (960,)
        col_flat = col_grid.reshape(-1)   # (960,)

        # 3. Assemble coordinate tensor (B, N, 3)
        row_batch = row_flat.unsqueeze(0).expand(B, -1)    # (B, 960)
        col_batch = col_flat.unsqueeze(0).expand(B, -1)    # (B, 960)
        coords = torch.stack([row_batch, col_batch, depth_flat], dim=-1)  # (B, 960, 3)

        # 4. Patch embedding returns (tokens, patch_resolution) tuple
        patch_tokens, patch_resolution = self.vit.patch_embed(x)  # (B, 960, embed_dim), (H_tok, W_tok)

        # 5. Standard pretrained 2D PE base + MLP depth correction
        # pos_embed is (1, 960, embed_dim) — no cls token since with_cls_token=False
        pe_base = self.vit.pos_embed                        # (1, 960, embed_dim)
        pe_correction = self.depth_cond_pe(coords)          # (B, 960, embed_dim)

        # pos_embed added here; vit.forward() is NOT called
        x_tokens = patch_tokens + pe_base + pe_correction   # (B, 960, embed_dim)

        # 6. Apply dropout after pos embed (matches vit.forward)
        x_tokens = self.vit.drop_after_pos(x_tokens)

        # 7. Apply pre_norm if it exists (identity in most configs)
        x_tokens = self.vit.pre_norm(x_tokens)

        # 8. Run through ViT transformer layers and final norm (ln1)
        for i, layer in enumerate(self.vit.layers):
            x_tokens = layer(x_tokens)
            if i == len(self.vit.layers) - 1 and self.vit.final_norm:
                x_tokens = self.vit.ln1(x_tokens)

        # 9. Reshape back to feature map: (B, embed_dim, H_tok, W_tok)
        feat = x_tokens.reshape(B, H_tok, W_tok, -1).permute(0, 3, 1, 2)
        return feat


# ── HEAD ──────────────────────────────────────────────────────────────────────

class Pose3DHead(nn.Module):
    """Transformer decoder head.

    Learnable joint queries (num_joints × hidden_dim) cross-attend to the
    flattened backbone feature map, then a per-joint linear predicts (x,y,z).

    Architecture:
        input_proj : Linear(in_channels → hidden_dim)
        joint_queries : Embedding(num_joints, hidden_dim)
        decoder : TransformerDecoder (num_layers layers, num_heads heads)
        joints_out : Linear(hidden_dim → 3)
    """

    def __init__(self, in_channels, num_joints=NUM_JOINTS, hidden_dim=256,
                 num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.num_joints   = num_joints
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
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.joint_queries.weight, std=0.02)
        for m in [self.joints_out, self.depth_out, self.uv_out]:
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        B = feat.size(0)
        # (B, C, H, W) → (B, H*W, hidden_dim)
        memory = self.input_proj(feat.flatten(2).transpose(1, 2))
        # (B, num_joints, hidden_dim)
        queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
        out = self.decoder(queries, memory)               # (B, num_joints, hidden_dim)
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
    """Backbone + Head wrapper. Input (B,4,H,W) → dict of joints/pelvis predictions."""

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
        return self.head(self.backbone(x))

    def load_pretrained(self, ckpt_path: str, verbose: bool = True) -> None:
        load_sapiens_pretrained(self, ckpt_path, verbose=verbose)
