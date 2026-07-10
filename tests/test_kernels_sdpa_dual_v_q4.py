"""Correctness tests for `sdpa_dual_v_q4` (quantized-input shared-QK dual-V SDPA).

Compared against `sdpa_dual_v_q4_reference`, which dequantizes the SAME
packed bits that the kernel reads, so quantization noise is shared across
both sides — the only diff is fp32 register accumulator vs fp16 reference
attention. Tolerance is set tight (atol/rtol = 5e-3 for fp16, 5e-2 for bf16).
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_motif.kernels import sdpa_dual_v_q4, sdpa_dual_v_q4_reference

_SHAPES = [
    # (B, H_q, H_kv, KV, d)
    (1, 4, 4, 16, 128),  # smallest
    (1, 4, 4, 128, 128),
    (1, 4, 4, 300, 128),  # not block-aligned (300 % BN=32 != 0)
    (1, 4, 4, 1024, 128),
    (2, 4, 4, 64, 128),  # batch
    # GQA cases — Motif uses gqa=5 for the 12.7B (40 Q heads / 8 KV heads).
    (1, 8, 4, 64, 128),  # gqa=2
    (1, 16, 4, 64, 128),  # gqa=4
    (1, 40, 8, 256, 128),  # Motif decode shape
    (1, 40, 8, 1024, 128),
]


@pytest.mark.parametrize("B,H_q,H_kv,KV,d", _SHAPES)
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [64])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_sdpa_dual_v_q4_matches_reference(B, H_q, H_kv, KV, d, bits, group_size, dtype):
    if d % group_size != 0:
        pytest.skip(f"d={d} not divisible by group_size={group_size}")

    mx.random.seed(0)
    scale = 1.0 / (d**0.5)
    q = mx.random.normal((B, H_q, 1, d)).astype(dtype)
    k = mx.random.normal((B, H_kv, KV, d)).astype(dtype)
    v1 = mx.random.normal((B, H_kv, KV, d)).astype(dtype)
    v2 = mx.random.normal((B, H_kv, KV, d)).astype(dtype)

    k_q = mx.quantize(k, group_size=group_size, bits=bits)
    v1_q = mx.quantize(v1, group_size=group_size, bits=bits)
    v2_q = mx.quantize(v2, group_size=group_size, bits=bits)

    a = sdpa_dual_v_q4(q, k_q, v1_q, v2_q, scale, group_size=group_size, bits=bits)
    b = sdpa_dual_v_q4_reference(q, k_q, v1_q, v2_q, scale, group_size=group_size, bits=bits)

    assert a.shape == (B, H_q, 1, 2 * d)
    assert a.dtype == dtype

    # bf16 tolerates more roundoff than fp16; the reference path uses MLX's
    # built-in SDPA whose accumulator differs from our fp32-in-registers path.
    atol = 5e-3 if dtype == mx.float16 else 5e-2
    rtol = 5e-3 if dtype == mx.float16 else 5e-2
    md = float(mx.max(mx.abs(a - b)))
    assert mx.allclose(a, b, atol=atol, rtol=rtol).item(), f"max diff = {md:.3e} (tol={atol})"


def test_sdpa_dual_v_q4_returns_concatenated_layout():
    """First D channels are attn·V1; second D channels are attn·V2.

    Verify by feeding V1 = V2: both halves must match exactly.
    Then feed V1 ≠ V2 and check the halves diverge.
    """
    mx.random.seed(0)
    B, H, KV, D = 1, 4, 64, 128
    scale = 1.0 / (D**0.5)
    q = mx.random.normal((B, H, 1, D)).astype(mx.float16)
    k = mx.random.normal((B, H, KV, D)).astype(mx.float16)
    v = mx.random.normal((B, H, KV, D)).astype(mx.float16)

    k_q = mx.quantize(k, group_size=64, bits=4)
    v_q = mx.quantize(v, group_size=64, bits=4)

    # Same V on both slabs ⇒ both halves must be identical.
    out_same = sdpa_dual_v_q4(q, k_q, v_q, v_q, scale, group_size=64, bits=4)
    assert out_same.shape == (B, H, 1, 2 * D)
    assert mx.allclose(out_same[..., :D], out_same[..., D:]).item()

    # Distinct V tensors ⇒ halves must diverge.
    v_other = mx.random.normal((B, H, KV, D)).astype(mx.float16)
    v_other_q = mx.quantize(v_other, group_size=64, bits=4)
    out_diff = sdpa_dual_v_q4(q, k_q, v_q, v_other_q, scale, group_size=64, bits=4)
    assert not mx.allclose(out_diff[..., :D], out_diff[..., D:], atol=1e-3).item()


@pytest.mark.parametrize(
    "B,H_q,H_kv,cap,kv_len,d",
    [
        (1, 8, 8, 256, 1, 128),  # decode step 1 into a step-padded buffer
        (1, 8, 8, 256, 100, 128),  # unaligned live length
        (1, 32, 8, 256, 100, 128),  # 12.7B GQA origin call shape
        (1, 40, 8, 512, 385, 128),  # capacity > one step
    ],
)
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_sdpa_dual_v_q4_runtime_kv_len_matches_sliced_reference(
    B, H_q, H_kv, cap, kv_len, d, bits, dtype
):
    """Full-capacity quantized triples + runtime kv_len must equal the
    reference computed on the exact-length live slice of the same packed
    bits (quantized-cache decode contract: full buffers + kv_len=offset)."""
    group_size = 64
    mx.random.seed(9)
    scale = 1.0 / (d**0.5)
    q = mx.random.normal((B, H_q, 1, d)).astype(dtype)
    # Poison the padding region so any read past kv_len diverges loudly.
    live = mx.random.normal((B, H_kv, kv_len, d)).astype(dtype)
    poison = mx.ones((B, H_kv, cap - kv_len, d), dtype=dtype) * 1e3
    full = mx.concatenate([live, poison], axis=2)

    k_q = mx.quantize(full, group_size=group_size, bits=bits)
    v1_q = mx.quantize(full * 0.5, group_size=group_size, bits=bits)
    v2_q = mx.quantize(full * -0.25, group_size=group_size, bits=bits)

    a = sdpa_dual_v_q4(q, k_q, v1_q, v2_q, scale, group_size=group_size, bits=bits, kv_len=kv_len)
    b = sdpa_dual_v_q4_reference(
        q, k_q, v1_q, v2_q, scale, group_size=group_size, bits=bits, kv_len=kv_len
    )

    assert a.shape == (B, H_q, 1, 2 * d)
    atol = 5e-3 if dtype == mx.float16 else 5e-2
    rtol = 5e-3 if dtype == mx.float16 else 5e-2
    md = float(mx.max(mx.abs(a - b)))
    assert mx.allclose(a, b, atol=atol, rtol=rtol).item(), (
        f"cap={cap} kv_len={kv_len} bits={bits}: max diff = {md:.3e} (tol={atol})"
    )


def test_sdpa_dual_v_q4_growing_kv_len_no_respecialization():
    """Growing kv_len over a fixed-capacity quantized buffer stays correct at
    every length — the loop bound is a runtime input, not a template."""
    B, H_q, H_kv, cap, d = 1, 32, 8, 256, 128
    group_size, bits = 64, 4
    mx.random.seed(13)
    scale = 1.0 / (d**0.5)
    q = mx.random.normal((B, H_q, 1, d)).astype(mx.float16)
    full = mx.random.normal((B, H_kv, cap, d)).astype(mx.float16)
    k_q = mx.quantize(full, group_size=group_size, bits=bits)
    v1_q = mx.quantize(full * 0.5, group_size=group_size, bits=bits)
    v2_q = mx.quantize(full * -0.25, group_size=group_size, bits=bits)

    for kv_len in (1, 31, 32, 33, 100, 255, 256):
        a = sdpa_dual_v_q4(
            q, k_q, v1_q, v2_q, scale, group_size=group_size, bits=bits, kv_len=kv_len
        )
        b = sdpa_dual_v_q4_reference(
            q, k_q, v1_q, v2_q, scale, group_size=group_size, bits=bits, kv_len=kv_len
        )
        md = float(mx.max(mx.abs(a - b)))
        assert mx.allclose(a, b, atol=5e-3, rtol=5e-3).item(), (
            f"kv_len={kv_len}: max diff = {md:.3e}"
        )


@pytest.mark.parametrize(
    "D,bits",
    [
        (256, 8),  # qk_per_thread=8 > EL_PER_INT=4
        (512, 4),  # qk_per_thread=16 > EL_PER_INT=8
    ],
)
def test_sdpa_dual_v_q4_rejects_out_of_contract_head_dim(D, bits):
    """Shapes violating the packed word contract (D/32 > 32/bits) must raise.

    The kernel loads exactly one packed uint32 per lane per tensor per step and
    unpacks D/32 channels from it. When D/32 > 32/bits the shift amounts exceed
    31 (undefined on a uint32 in Metal) and half the channels come from a word
    that is never loaded, so the kernel would return silent numerical garbage.
    The wrapper must fail loudly instead. group_size divides D, so only the
    D/bits relation is under test here.
    """
    group_size = 128  # divides D=256/512 and satisfies the group asserts
    B, H, KV = 1, 4, 64
    scale = 1.0 / (D**0.5)
    q = mx.random.normal((B, H, 1, D)).astype(mx.float16)
    k = mx.random.normal((B, H, KV, D)).astype(mx.float16)
    v = mx.random.normal((B, H, KV, D)).astype(mx.float16)

    k_q = mx.quantize(k, group_size=group_size, bits=bits)
    v_q = mx.quantize(v, group_size=group_size, bits=bits)

    with pytest.raises(AssertionError, match=r"D/32 <= 32/bits"):
        out = sdpa_dual_v_q4(q, k_q, v_q, v_q, scale, group_size=group_size, bits=bits)
        mx.eval(out)


@pytest.mark.parametrize(
    "D,bits",
    [
        # qk_per_thread = D/32 satisfies the OLD guard (D/32 <= 32/bits) but does
        # NOT divide EL_PER_INT = 32/bits, so a lane's contiguous channel block
        # straddles two packed uint32 words. These are the shapes the necessary-
        # only guard let through; the sufficient guard must reject them.
        (96, 8),  # qk_per_thread=3, EL_PER_INT=4: 3<=4 (old OK) but 3 does not divide 4
        (96, 4),  # qk_per_thread=3, EL_PER_INT=8: 3<=8 (old OK) but 3 does not divide 8
        (160, 4),  # qk_per_thread=5, EL_PER_INT=8: 5<=8 (old OK) but 5 does not divide 8
    ],
)
def test_sdpa_dual_v_q4_rejects_word_straddling_head_dim(D, bits):
    """Non-power-of-two head dims whose lane block straddles a packed word.

    D/32 <= 32/bits (the old NECESSARY guard) is not enough: unless
    qk_per_thread | EL_PER_INT the block [base, base+qk_per_thread) crosses a
    uint32 boundary for some lane, so channels past the boundary are read from a
    word that was never loaded and their shifts reach >= 32 bits (UB in Metal).
    The sufficient guard must fail loudly for these instead of returning garbage.
    """
    group_size = 32  # divides D=96 and D=160; isolates the word-straddle condition
    assert D % group_size == 0
    assert D // 32 <= 32 // bits, "test shape must PASS the old necessary-only guard"
    B, H, KV = 1, 4, 64
    scale = 1.0 / (D**0.5)
    q = mx.random.normal((B, H, 1, D)).astype(mx.float16)
    k = mx.random.normal((B, H, KV, D)).astype(mx.float16)
    v = mx.random.normal((B, H, KV, D)).astype(mx.float16)

    k_q = mx.quantize(k, group_size=group_size, bits=bits)
    v_q = mx.quantize(v, group_size=group_size, bits=bits)

    with pytest.raises(AssertionError, match=r"qk_per_thread \| EL_PER_INT"):
        out = sdpa_dual_v_q4(q, k_q, v_q, v_q, scale, group_size=group_size, bits=bits)
        mx.eval(out)


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("bits", [4, 8])
def test_sdpa_dual_v_q4_accepts_shipped_shape_boundaries(group_size, bits):
    """D=128 (the shipped head dim) must PASS the sufficient guard for every
    supported group_size (32/64/128) and bits (4/8): qk_per_thread=4 divides both
    EL_PER_INT in {4,8} and every group_size in {32,64,128}. The kernel runs and
    matches the dequant reference (no AssertionError, correct numerics).
    """
    D = 128
    B, H, KV = 1, 4, 64
    mx.random.seed(0)
    scale = 1.0 / (D**0.5)
    q = mx.random.normal((B, H, 1, D)).astype(mx.float16)
    k = mx.random.normal((B, H, KV, D)).astype(mx.float16)
    v1 = mx.random.normal((B, H, KV, D)).astype(mx.float16)
    v2 = mx.random.normal((B, H, KV, D)).astype(mx.float16)

    k_q = mx.quantize(k, group_size=group_size, bits=bits)
    v1_q = mx.quantize(v1, group_size=group_size, bits=bits)
    v2_q = mx.quantize(v2, group_size=group_size, bits=bits)

    a = sdpa_dual_v_q4(q, k_q, v1_q, v2_q, scale, group_size=group_size, bits=bits)
    b = sdpa_dual_v_q4_reference(q, k_q, v1_q, v2_q, scale, group_size=group_size, bits=bits)

    assert a.shape == (B, H, 1, 2 * D)
    md = float(mx.max(mx.abs(a - b)))
    assert mx.allclose(a, b, atol=5e-3, rtol=5e-3).item(), (
        f"group_size={group_size} bits={bits}: max diff = {md:.3e}"
    )
