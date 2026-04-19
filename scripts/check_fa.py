"""Verify which SDPA backend fires for our decoder SA and CA shapes."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from model.motion_decoder import MultiheadSDPA, build_block_staircase_mask


def try_backend(name: str, backend: SDPBackend, fn) -> str:
    try:
        with sdpa_kernel([backend]):
            fn()
        return f"{name:16s}  ✓ runs"
    except Exception as e:  # noqa: BLE001
        return f"{name:16s}  ✗ {type(e).__name__}: {str(e)[:90]}"


def main() -> None:
    device = "cuda"
    B, N, T, d, heads = 16, 1, 16, 256, 8
    latents = 192

    sa = MultiheadSDPA(d, heads).to(device).eval()
    ca = MultiheadSDPA(d, heads).to(device).eval()

    x = torch.randn(B, N * T, d, device=device, dtype=torch.bfloat16)
    kv = torch.randn(B, latents, d, device=device, dtype=torch.bfloat16)
    mask = build_block_staircase_mask(N, T, torch.device(device))

    # cast modules' params to bf16 for backend test
    sa = sa.bfloat16()
    ca = ca.bfloat16()

    print("=== Decoder Self-Attn (with block-staircase bool mask) ===")
    for name, be in [("flash", SDPBackend.FLASH_ATTENTION),
                     ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
                     ("math", SDPBackend.MATH)]:
        print(try_backend(name, be, lambda: sa(x, x, x, attn_mask=mask)))

    print("\n=== Decoder Cross-Attn (no mask) ===")
    for name, be in [("flash", SDPBackend.FLASH_ATTENTION),
                     ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
                     ("math", SDPBackend.MATH)]:
        print(try_backend(name, be, lambda: ca(x, kv, kv)))

    print("\n=== Encoder Perceiver CA-shaped (B=16, Q=192, KV=1136) ===")
    q = torch.randn(B, 192, d, device=device, dtype=torch.bfloat16)
    scene = torch.randn(B, 1136, d, device=device, dtype=torch.bfloat16)
    perc = MultiheadSDPA(d, heads).to(device).bfloat16().eval()
    for name, be in [("flash", SDPBackend.FLASH_ATTENTION),
                     ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
                     ("math", SDPBackend.MATH)]:
        print(try_backend(name, be, lambda: perc(q, scene, scene)))


if __name__ == "__main__":
    main()
