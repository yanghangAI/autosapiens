"""Sapiens-based RGBD pose estimation model — design003: near_emphasized_continuous_depth_pe.

Replaces the standard 2D positional embedding in the ViT backbone with a
decomposed row + column + near-emphasized continuously interpolated depth
positional embedding.
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
from mmpretrain.models.utils import resize_pos_embed

from infra import NUM_JOINTS, SAPIENS_ARCHS, _interp_pos_embed


# ── DEPTH BUCKET PE ───────────────────────────────────────────────────────────

class DepthBucketPE(nn.Module):
    """Decomposed row + column + near-emphasized continuous depth PE.

    Replaces the ViT's standard 2D pos_embed.
    Parameters:
        row_emb  : (H_tok, embed_dim)  — init from pretrained row-mean
        col_emb  : (W_tok, embed_dim)  — init from pretrained col-mean
        depth_emb: (num_depth_bins, embed_dim) — zero-initialized anchors
    """

    def __init__(self, h_tok: int, w_tok: int, embed_dim: int, num_depth_bins: int = 16):
        super().__init__()
        self.h_tok = h_tok
        self.w_tok = w_tok
        self.num_depth_bins = num_depth_bins

        self.row_emb   = nn.Parameter(torch.zeros(h_tok, embed_dim))
        self.col_emb   = nn.Parameter(torch.zeros(w_tok, embed_dim))
        self.depth_emb = nn.Parameter(torch.zeros(num_depth_bins, embed_dim))

    def forward(self, patch_tokens: torch.Tensor, depth_ch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, N, D) where N = h_tok * w_tok
            depth_ch:     (B, 1, H_img, W_img) — normalized depth in [0,1]
        Returns:
            (B, N, D) patch_tokens + composed PE
        """
        B = patch_tokens.shape[0]
        h, w = self.h_tok, self.w_tok
        device = patch_tokens.device

        # Row / col indices
        rows = torch.arange(h, device=device).unsqueeze(1).expand(h, w).reshape(-1)  # (N,)
        cols = torch.arange(w, device=device).unsqueeze(0).expand(h, w).reshape(-1)  # (N,)

        # Average-pool depth to the patch grid, then remap with sqrt so
        # smaller depths get denser effective anchor coverage before
        # interpolating between neighboring learned depth anchors.
        depth_patches = F.avg_pool2d(depth_ch, kernel_size=16, stride=16)  # (B, 1, h, w)
        depth_norm = depth_patches.squeeze(1).clamp(0.0, 1.0)
        depth_pos = torch.sqrt(depth_norm) * (self.num_depth_bins - 1)
        idx_lo = torch.floor(depth_pos).long()
        idx_hi = torch.clamp(idx_lo + 1, max=self.num_depth_bins - 1)
        alpha = (depth_pos - idx_lo.float()).reshape(B, -1, 1)

        depth_lo = self.depth_emb[idx_lo.reshape(B, -1)]
        depth_hi = self.depth_emb[idx_hi.reshape(B, -1)]
        depth_pe = (1.0 - alpha) * depth_lo + alpha * depth_hi

        # Compose PE: (1, N, D) + (1, N, D) + (B, N, D)
        pe = (self.row_emb[rows].unsqueeze(0)        # (1, N, D)
              + self.col_emb[cols].unsqueeze(0)       # (1, N, D)
              + depth_pe)                             # (B, N, D)

        return patch_tokens + pe


# ── BACKBONE ──────────────────────────────────────────────────────────────────

class SapiensBackboneRGBD(nn.Module):
    """Sapiens ViT with n-channel patch embed + decomposed depth bucket PE.

    For design001 (8-channel stacking), in_channels=8:
    channels are [RGB_t(0:3), D_t(3), RGB_{t-5}(4:7), D_{t-5}(7)].
    """

    def __init__(self, arch="sapiens_0.3b", img_size=(384, 640), drop_path_rate=0.0,
                 num_depth_bins: int = 16, in_channels: int = 4):
        super().__init__()
        if arch not in SAPIENS_ARCHS:
            raise ValueError(f"Unknown arch '{arch}'. Choose from {list(SAPIENS_ARCHS)}")
        self.arch      = arch
        self.embed_dim = SAPIENS_ARCHS[arch]["embed_dim"]
        self.in_channels = in_channels

        self.vit = VisionTransformer(
            arch=arch, img_size=img_size, patch_size=16, in_channels=in_channels,
            qkv_bias=True, final_norm=True, drop_path_rate=drop_path_rate,
            with_cls_token=False, out_type="featmap", patch_cfg=dict(padding=2),
        )

        # Infer patch grid dimensions
        h_tok, w_tok = self.vit.patch_resolution  # (40, 24) for 640×384
        self.h_tok = h_tok
        self.w_tok = w_tok

        # Custom decomposed PE
        self.depth_bucket_pe = DepthBucketPE(h_tok, w_tok, self.embed_dim, num_depth_bins)

    def _run_vit_manual(self, x: torch.Tensor, depth_ch: torch.Tensor) -> torch.Tensor:
        """Manually execute ViT forward with custom PE injection.

        Replicates VisionTransformer.forward but replaces the pos_embed addition
        with our DepthBucketPE.
        """
        vit = self.vit
        B = x.shape[0]

        # Patch embedding → (B, N, D)
        patch_tokens, patch_resolution = vit.patch_embed(x)

        # No cls_token (with_cls_token=False), so skip cls token concat

        # Add our decomposed PE (instead of vit.pos_embed which is zeroed buffer)
        patch_tokens = self.depth_bucket_pe(patch_tokens, depth_ch)

        # drop_after_pos (likely Identity or Dropout — still apply it)
        patch_tokens = vit.drop_after_pos(patch_tokens)

        # pre_norm (Identity when pre_norm=False)
        patch_tokens = vit.pre_norm(patch_tokens)

        # Run transformer layers
        outs = []
        for i, layer in enumerate(vit.layers):
            patch_tokens = layer(patch_tokens)
            if i == len(vit.layers) - 1 and vit.final_norm:
                patch_tokens = vit.ln1(patch_tokens)
            if i in vit.out_indices:
                outs.append(vit._format_output(patch_tokens, patch_resolution))

        return outs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        depth_ch = x[:, 3:4, :, :]          # (B, 1, H, W) — normalized depth
        outs = self._run_vit_manual(x, depth_ch)
        return outs[0]                        # (B, embed_dim, H_tok, W_tok)


# ── HEAD ──────────────────────────────────────────────────────────────────────

class Pose3DHead(nn.Module):
    """Transformer decoder head (unchanged from baseline)."""

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
        self.depth_out  = nn.Linear(hidden_dim, 1)
        self.uv_out     = nn.Linear(hidden_dim, 2)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.joint_queries.weight, std=0.02)
        for m in [self.joints_out, self.depth_out, self.uv_out]:
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        B = feat.size(0)
        memory  = self.input_proj(feat.flatten(2).transpose(1, 2))
        queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
        out = self.decoder(queries, memory)
        pelvis_token = out[:, 0, :]
        return {
            "joints":       self.joints_out(out),
            "pelvis_depth": self.depth_out(pelvis_token),
            "pelvis_uv":    self.uv_out(pelvis_token),
        }


# ── WEIGHT LOADING ────────────────────────────────────────────────────────────

def load_sapiens_pretrained(model: nn.Module, ckpt_path: str, verbose: bool = True) -> None:
    """Load Sapiens RGB pretrained checkpoint.

    - Expands patch embed 3→4 channels
    - Interpolates pos_embed, then uses it to initialize DepthBucketPE
      (row_emb = row-mean, col_emb = col-mean), and zeros out vit.pos_embed
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    src = raw.get("state_dict", raw.get("model", raw))

    remapped = {f"backbone.vit.{k}": v for k, v in src.items()}
    model_sd = model.state_dict()

    # 1. Expand patch embed 3 → 4 channels, then 4 → 8 channels if needed
    pe_key = "backbone.vit.patch_embed.projection.weight"
    if pe_key in remapped:
        w_rgb = remapped[pe_key]
        # 3 → 4 channels: add depth channel as mean of RGB channels
        w_4ch = torch.cat([w_rgb, w_rgb.mean(dim=1, keepdim=True)], dim=1)
        if verbose:
            print("[weights] patch_embed: 3ch → 4ch  (depth = mean of RGB channels)")
        # 4 → 8 channels: add 4 temporal channels as mean of existing 4 channels
        n_ch = model_sd[pe_key].shape[1]
        if n_ch == 8:
            w_8ch = torch.cat([w_4ch, w_4ch.mean(dim=1, keepdim=True).expand(-1, 4, -1, -1)], dim=1)
            remapped[pe_key] = w_8ch
            if verbose:
                print("[weights] patch_embed: 4ch → 8ch  (temporal channels = mean of 4 original channels)")
        else:
            remapped[pe_key] = w_4ch

    # 2. Interpolate pos_embed and extract row/col for DepthBucketPE init
    pe_key = "backbone.vit.pos_embed"
    pretrained_pe_2d = None
    if pe_key in remapped and pe_key in model_sd:
        src_pe = remapped[pe_key]
        try:
            tgt_h, tgt_w = model.backbone.vit.patch_resolution
        except AttributeError:
            tgt_N = model_sd[pe_key].shape[1]
            tgt_h = tgt_w = int(tgt_N ** 0.5)
        src_g = int((src_pe.shape[1] - 1) ** 0.5)
        interp_pe = _interp_pos_embed(src_pe, tgt_h, tgt_w, has_cls=True).to(src_pe.dtype)
        # interp_pe: (1, tgt_h*tgt_w, 1024)
        pretrained_pe_2d = interp_pe.squeeze(0).reshape(tgt_h, tgt_w, -1)  # (H, W, D)
        if verbose:
            print(f"[weights] pos_embed: {src_g}×{src_g} → {tgt_h}×{tgt_w}  (bicubic)")
        # Replace pos_embed entry with zeros — it's now a buffer, not a parameter
        # Do NOT load it into the model (skip it below)
        del remapped[pe_key]

    remapped.pop("backbone.vit.cls_token", None)

    load_sd, missing, shape_mismatch = {}, [], []
    for k, v_model in model_sd.items():
        if not k.startswith("backbone."):
            continue
        # Skip depth_bucket_pe (will be initialized below)
        if "depth_bucket_pe" in k:
            continue
        # Skip pos_embed (now a zero buffer)
        if k == "backbone.vit.pos_embed":
            continue
        if k not in remapped:
            missing.append(k); continue
        if v_model.shape != remapped[k].shape:
            shape_mismatch.append(f"  {k}: model {v_model.shape} vs ckpt {remapped[k].shape}"); continue
        load_sd[k] = remapped[k]

    model.load_state_dict(load_sd, strict=False)

    # 3. Initialize DepthBucketPE from pretrained 2D pos_embed
    if pretrained_pe_2d is not None:
        dpe = model.backbone.depth_bucket_pe
        row_init = pretrained_pe_2d.mean(dim=1)  # mean over cols → (H, D)
        col_init = pretrained_pe_2d.mean(dim=0)  # mean over rows → (W, D)
        with torch.no_grad():
            dpe.row_emb.data.copy_(row_init)
            dpe.col_emb.data.copy_(col_init)
            # depth_emb stays zero (already initialized to zero)
        if verbose:
            print(f"[weights] DepthBucketPE: row_emb & col_emb init from pretrained 2D PE; "
                  f"depth_emb zero-initialized")

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
    """Backbone + Head wrapper.

    For design001: Input (B, 8, H, W) → dict of joints/pelvis predictions.
    Channels: [RGB_t(0:3), D_t(3), RGB_{t-5}(4:7), D_{t-5}(7)].
    """

    def __init__(self, arch="sapiens_0.3b", img_size=(640, 384), num_joints=NUM_JOINTS,
                 head_hidden=256, head_num_heads=8, head_num_layers=4,
                 head_dropout=0.1, drop_path_rate=0.0, num_depth_bins=16,
                 in_channels=4):
        super().__init__()
        self.backbone = SapiensBackboneRGBD(arch=arch, img_size=img_size,
                                             drop_path_rate=drop_path_rate,
                                             num_depth_bins=num_depth_bins,
                                             in_channels=in_channels)

        # After building the ViT, zero out vit.pos_embed and register it as a buffer
        # so it does not appear as a trainable parameter in the backbone group
        vit = self.backbone.vit
        pos_embed_shape = vit.pos_embed.shape
        del vit.pos_embed
        vit.register_buffer("pos_embed", torch.zeros(pos_embed_shape))

        self.head = Pose3DHead(in_channels=SAPIENS_ARCHS[arch]["embed_dim"],
                               num_joints=num_joints, hidden_dim=head_hidden,
                               num_heads=head_num_heads, num_layers=head_num_layers,
                               dropout=head_dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.backbone(x))

    def load_pretrained(self, ckpt_path: str, verbose: bool = True) -> None:
        load_sapiens_pretrained(self, ckpt_path, verbose=verbose)
