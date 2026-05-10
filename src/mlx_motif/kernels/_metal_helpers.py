"""Shared Metal source helpers reused across q4 kernel modules."""

from __future__ import annotations

# 4-bit-only specialisations of MLX's `load_vector` and `qdot` from
# mlx/backend/metal/kernels/quantized.h.  We duplicate rather than `#include`
# because mx.fast.metal_kernel doesn't expose MLX's internal headers to user
# kernels.  Any kernel that needs these helpers should pass this string as the
# `header` argument to `mx.fast.metal_kernel`.
Q4_QDOT_HEADER = r"""
template <typename T, typename U, int values_per_thread>
inline U load_vector_q4(const device T* x, thread U* x_thread) {
    U sum = 0;
    for (int i = 0; i < values_per_thread; i += 4) {
        sum += float(x[i]) + float(x[i+1]) + float(x[i+2]) + float(x[i+3]);
        x_thread[i  ] = float(x[i]);
        x_thread[i+1] = float(x[i+1]) / 16.0f;
        x_thread[i+2] = float(x[i+2]) / 256.0f;
        x_thread[i+3] = float(x[i+3]) / 4096.0f;
    }
    return sum;
}

template <typename U, int values_per_thread>
inline U qdot_q4(const device uint8_t* w,
                 const thread U* x_thread,
                 U scale, U bias, U sum) {
    U accum = 0;
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
        accum += (x_thread[4*i  ] * float(ws[i] & 0x000fu) +
                  x_thread[4*i+1] * float(ws[i] & 0x00f0u) +
                  x_thread[4*i+2] * float(ws[i] & 0x0f00u) +
                  x_thread[4*i+3] * float(ws[i] & 0xf000u));
    }
    return scale * accum + sum * bias;
}
"""
