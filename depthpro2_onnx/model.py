"""Focal-free Depth Pro 2 graphs for export: RGB in [0, 1] plus a known focal length in, meters out.

``StreamDepth`` follows the unbuffered streaming step of Apple's Core ML exporter
(``demo/export/stream.py`` in ml-depth-pro-2), with its caches passed in and returned as tensors.
"""

from __future__ import annotations

import torch
from depth_pro_2 import DepthPro2
from depth_pro_2.model import IMAGENET_MEAN, IMAGENET_STD, INPUT_MULTIPLE
from depth_pro_2.network.encoder import eva_finish, eva_qkv_rope
from torch import nn
from torch.nn import functional as F

# Hub revisions the released graphs were exported from.
CHECKPOINTS = {
    "S": ("apple/DepthPro2-S", "30e647174773e3bdf5ab5dcc2a815c38cec5fdd8"),
    "B": ("apple/DepthPro2-B", "281dc9883dea8fba21d5d78734291ca5d4f159d6"),
}
FOCAL_CHECKPOINT = ("apple/DepthPro2-focal", "f1be7a9715938d76c687554156357f0492266f02")

# Added to the logits of cache slots no frame has filled yet; exp() of it is 0 in float16 too.
EMPTY_SLOT_LOGIT = -3.0e4
# Blocks whose attention logits exceed float16's range on real images (|logit| > 2e6 in block 0
# of S, 2e5 of B); they keep float32 attention in the float16 graphs.
FP32_ATTENTION_BLOCKS = (0,)


def load_model(
    size: str, device: str | torch.device = "cpu", precision: torch.dtype = torch.float32
) -> DepthPro2:
    """Depth Pro 2 ``size`` ("S" or "B") without the field-of-view network, in eval mode."""
    repo, revision = CHECKPOINTS[size]
    model = DepthPro2.from_pretrained(repo, revision=revision, precision=precision)
    if model.fov is not None:
        raise RuntimeError(f"{repo}@{revision} unexpectedly bundles the focal-length network.")
    return model.to(device).eval()


def check_size(height: int, width: int) -> None:
    # The released graphs skip Apple's replicate padding: padding changes the depth, so callers
    # must crop or resize to a multiple of 32 themselves and adjust f_px to match.
    if height % INPUT_MULTIPLE or width % INPUT_MULTIPLE:
        raise ValueError(f"{height}x{width} is not a multiple of {INPUT_MULTIPLE}.")


class Fp32Attention(nn.Module):
    """Runs a timm ``EvaAttention`` in float32 inside an otherwise float16 network."""

    def __init__(self, attn: nn.Module):
        super().__init__()
        self.attn = attn.float()

    def forward(self, x, rope=None, attn_mask=None, is_causal=False):
        rope = None if rope is None else rope.float()
        attn_mask = None if attn_mask is None else attn_mask.float()
        out = self.attn(x.float(), rope=rope, attn_mask=attn_mask, is_causal=is_causal)
        return out.to(x.dtype)


def to_fp16(model: DepthPro2) -> DepthPro2:
    """Cast ``model`` to float16 in place, except the parts that overflow or lose precision."""
    model.half()
    # DepthPro2 runs its head in float32 by design.
    model.head.float()
    # RoPE tables are computed at export time; float16 periods would blur patch positions.
    model.encoder.rope.float()
    for idx in FP32_ATTENTION_BLOCKS:
        block = model.encoder.blocks[idx]
        block.attn = Fp32Attention(block.attn)
    return model


class _Normalize(nn.Module):
    def __init__(self, model: DepthPro2):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @property
    def dtype(self) -> torch.dtype:
        return self.model.encoder.patch_embed.proj.weight.dtype

    def normalize(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return ((pixel_values - self.mean) / self.std).to(self.dtype)


def to_meters(inverse_depth: torch.Tensor, f_px: torch.Tensor, width: int) -> torch.Tensor:
    # Float32 so that far points (inverse depth near 0) do not round to 0 or overflow.
    return f_px / (width * inverse_depth.float())


class ClipDepth(_Normalize):
    """Frames ``(T, 3, H, W)`` RGB in [0, 1] and ``f_px`` ``(1,)`` to depth ``(T, H, W)`` in meters.

    With T > 1 the frames attend to each other as one clip, as ``DepthPro2.infer`` does.
    """

    def forward(self, pixel_values: torch.Tensor, f_px: torch.Tensor) -> torch.Tensor:
        inverse_depth = self.model(self.normalize(pixel_values))[:, 0]
        return to_meters(inverse_depth, f_px, pixel_values.shape[-1])


class StreamDepth(_Normalize):
    """One frame of an unbuffered stream: the frame attends to itself and ``window - 1`` before it.

    Inputs are the frame ``(1, 3, H, W)`` RGB in [0, 1], ``f_px`` ``(1,)`` and the state returned
    by the previous call (zeros for the first frame, see ``state_shapes``). Outputs are the depth
    ``(1, H, W)`` in meters and the next state, in the order of ``state_names``.
    """

    def __init__(self, model: DepthPro2, window: int, height: int, width: int):
        super().__init__(model)
        if window < 2:
            raise ValueError("A stream needs a window of at least 2 frames; use ClipDepth for 1.")
        check_size(height, width)
        encoder = model.encoder
        self.window, self.height, self.width = window, height, width
        self.grid = (height // encoder.patch_size, width // encoder.patch_size)
        self.num_patches = self.grid[0] * self.grid[1]
        self.cross_blocks = sorted(encoder.cross_frame_blocks)
        self.register_buffer("rope", encoder.rope.get_embed(shape=self.grid), persistent=False)

    @property
    def state_names(self) -> list[str]:
        names = ["filled"]
        for idx in self.cross_blocks:
            names += [f"keys_{idx}", f"values_{idx}", f"prefixes_{idx}"]
        return names

    def state_shapes(self) -> dict[str, tuple]:
        encoder = self.model.encoder
        heads = encoder.blocks[self.cross_blocks[0]].attn.num_heads
        dim, npt = encoder.embed_dim, encoder.num_prefix_tokens
        kv = (1, heads, self.window * self.num_patches, dim // heads)
        shapes = {"filled": (self.window,)}
        for idx in self.cross_blocks:
            prefixes = (self.window, npt, dim)
            shapes |= {f"keys_{idx}": kv, f"values_{idx}": kv, f"prefixes_{idx}": prefixes}
        return shapes

    def initial_state(self, device=None) -> list[torch.Tensor]:
        return [
            torch.zeros(shape, dtype=self.dtype, device=device)
            for shape in self.state_shapes().values()
        ]

    def forward(self, pixel_values: torch.Tensor, f_px: torch.Tensor, *state: torch.Tensor):
        encoder = self.model.encoder
        npt = encoder.num_prefix_tokens
        filled, caches = state[0], state[1:]

        patches = encoder.patch_embed(self.normalize(pixel_values))
        prefix_tokens = torch.cat([encoder.cls_token, encoder.reg_token], dim=1)
        tokens = torch.cat([prefix_tokens, patches.flatten(1, 2)], dim=1)

        filled = torch.cat([filled[1:], torch.ones_like(filled[:1])])
        empty = ((1 - filled) * EMPTY_SLOT_LOGIT)[:, None].expand(-1, self.num_patches)
        mask = torch.cat([filled.new_zeros(npt), empty.flatten()])[None]
        prefix_weights = (filled / filled.sum()).view(-1, 1, 1)
        rope = self.rope

        new_state = [filled]
        features = []
        for idx, block in enumerate(encoder.blocks):
            if idx in encoder.cross_frame_blocks:
                slot = 3 * self.cross_blocks.index(idx)
                keys, values, prefixes = caches[slot : slot + 3]
                prefix, own = tokens[:, :npt], tokens[:, npt:]
                query, key, value = eva_qkv_rope(block, own, rope, 0)
                keys = torch.cat([keys[:, :, self.num_patches :], key], dim=2)
                values = torch.cat([values[:, :, self.num_patches :], value], dim=2)
                prefixes = torch.cat([prefixes[1:], prefix])
                pooled = (prefixes * prefix_weights).sum(dim=0, keepdim=True)
                _, pooled_key, pooled_value = eva_qkv_rope(block, pooled, None, npt)
                attended = F.scaled_dot_product_attention(
                    query,
                    torch.cat([pooled_key, keys], dim=2),
                    torch.cat([pooled_value, values], dim=2),
                    attn_mask=mask.to(query.dtype),
                )
                tokens = torch.cat([prefix, eva_finish(block, own, attended)], dim=1)
                new_state += [keys, values, prefixes]
            else:
                tokens = block(tokens, rope=rope)
            if idx in encoder.reassembled_blocks:
                features.append(tokens[:, npt:])

        pyramid = encoder.reassemble_tokens(features, self.grid)
        inverse_depth = self.model.head(self.model.decoder(pyramid).float())[:, 0]
        return (to_meters(inverse_depth, f_px, self.width), *new_state)
