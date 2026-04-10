"""Sapiens-based RGBD pose estimation model -- design005: Alternating Layer Average.

Extracts every-other ViT block (odd 0-indexed: 1,3,5,...,23) and averages them.
Zero new parameters.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SAPIENS_ROOT = Path("/work/pi_nwycoff_umass_edu/hang/auto/sapiens")
for _p in [str(_SAPIENS_ROOT / "pretrain"), str(_SAPIENS_ROOT / "engine")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

from mmpretrain.models.backbones.vision_transformer import VisionTransformer, resize_pos_embed

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

    def _run_vit_preamble(self, x: torch.Tensor):
        x, patch_resolution = self.vit.patch_embed(x)
        x = x + resize_pos_embed(
            self.vit.pos_embed,
            self.vit.patch_resolution,
            patch_resolution,
            mode=self.vit.interpolate_mode,
            num_extra_tokens=self.vit.num_extra_tokens)
        x = self.vit.drop_after_pos(x)
        x = self.vit.pre_norm(x)
        return x, patch_resolution

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract every-other block (odd 0-indexed), running-sum average."""
        x, patch_resolution = self._run_vit_preamble(x)
        H, W = patch_resolution

        extract_indices = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23}
        running_sum = None
        count = 0
        for i, layer in enumerate(self.vit.layers):
            x = layer(x)
            if i in extract_indices:
                normed = self.vit.ln1(x)
                if running_sum is None:
                    running_sum = normed
                else:
                    running_sum = running_sum + normed
                count += 1

        averaged = running_sum / count  # (B, N, C)

        B, N, C = averaged.shape
        return averaged.transpose(1, 2).reshape(B, C, H, W)


# ── HEAD ──────────────────────────────────────────────────────────────────────

class Pose3DHead(nn.Module):
    """Transformer decoder head."""

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
        memory = self.input_proj(feat.flatten(2).transpose(1, 2))
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
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    src = raw.get("state_dict", raw.get("model", raw))

    remapped = {f"backbone.vit.{k}": v for k, v in src.items()}
    model_sd = model.state_dict()

    pe_key = "backbone.vit.patch_embed.projection.weight"
    if pe_key in remapped:
        w_rgb = remapped[pe_key]
        remapped[pe_key] = torch.cat([w_rgb, w_rgb.mean(dim=1, keepdim=True)], dim=1)
        if verbose:
            print("[weights] patch_embed: 3ch -> 4ch  (depth = mean of RGB channels)")

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
            print(f"[weights] pos_embed: {src_g}x{src_g} -> {tgt_h}x{tgt_w}  (bicubic)")

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
    """Backbone + Head wrapper. No aggregator needed (averaging done in backbone)."""

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
