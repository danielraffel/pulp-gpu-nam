#pragma once

// Block kernels shared by the streaming CPU engines (WaveNet, LSTM): small dense
// matrix products over a run of frames, and array-wide activations.
//
// Frames are stored time-major: frame t of a C-channel signal is the C floats at
// x + t*C. A layer's weights are stored transposed, wt[j*Z + z] = W[z][j], so the
// inner loop runs over contiguous output channels and vectorizes without any
// horizontal reduction. Frames are processed in register-blocked groups so each
// weight column is loaded once per group rather than once per frame.
//
// Every routine here is allocation-free; callers own all buffers.

#include <cmath>
#include <cstddef>
#include <cstdint>

#ifndef GPU_NAM_HAVE_ACCELERATE
#define GPU_NAM_HAVE_ACCELERATE 0
#endif
#if GPU_NAM_HAVE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

namespace pulp::examples::nam {

// How the engines evaluate tanh (and, for LSTM gates, sigmoid).
//   Exact       std::tanh and 1/(1+std::exp(-x)), element by element: the libm
//               results a scalar implementation produces. The default.
//   Vector      the same functions array-wide through Accelerate's vForce on
//               Apple (within a couple of ulps of libm, several times faster);
//               identical to Exact where vForce is unavailable. Recurrent
//               models amplify those ulps, so it is opt-in rather than default.
//   NamRational the rational tanh approximation NAM players use when their
//               fast-tanh switch is on: odd, exact at 0, abs error <= 4.4e-4 for
//               |x| <= 10 (worst near |x| = 0.58), overshooting to 1.0081 as
//               |x| -> inf. Selecting it reproduces such a player's output; it is
//               a different function, not a better tanh.
enum class TanhMode { Exact, Vector, NamRational };

namespace kernels {

// The rational tanh approximation from NeuralAmpModelerCore's activations.h
// (MIT, Copyright (c) 2023 Steven Atkinson). Coefficients and evaluation order
// are kept verbatim so results match that implementation to rounding.
inline float nam_rational_tanh(float x) {
    const float ax = std::fabs(x);
    const float x2 = x * x;
    return (x * (2.45550750702956f + 2.45550750702956f * ax + (0.893229853513558f + 0.821226666969744f * ax) * x2)
            / (2.44506634652299f + (2.44506634652299f + x2) * std::fabs(x + 0.814642734961073f * x * ax)));
}

inline float nam_rational_sigmoid(float x) { return 0.5f * (nam_rational_tanh(x * 0.5f) + 1.0f); }

inline void tanh_inplace(float* d, std::size_t n, TanhMode mode) {
    if (mode == TanhMode::NamRational) {
        for (std::size_t i = 0; i < n; ++i) d[i] = nam_rational_tanh(d[i]);
        return;
    }
#if GPU_NAM_HAVE_ACCELERATE
    if (mode == TanhMode::Vector) {
        const int nn = static_cast<int>(n);
        vvtanhf(d, d, &nn);
        return;
    }
#endif
    for (std::size_t i = 0; i < n; ++i) d[i] = std::tanh(d[i]);
}

// Logistic sigmoid 1/(1+exp(-x)); the NamRational variant is the matching
// 0.5*(tanh(x/2)+1) built on the rational tanh.
inline void sigmoid_inplace(float* d, std::size_t n, TanhMode mode) {
    if (mode == TanhMode::NamRational) {
        for (std::size_t i = 0; i < n; ++i) d[i] = nam_rational_sigmoid(d[i]);
        return;
    }
#if GPU_NAM_HAVE_ACCELERATE
    if (mode == TanhMode::Vector) {
        for (std::size_t i = 0; i < n; ++i) d[i] = -d[i];
        const int nn = static_cast<int>(n);
        vvexpf(d, d, &nn);
        for (std::size_t i = 0; i < n; ++i) d[i] = 1.0f / (1.0f + d[i]);
        return;
    }
#endif
    for (std::size_t i = 0; i < n; ++i) d[i] = 1.0f / (1.0f + std::exp(-d[i]));
}

// Dense product over a run of frames with a fused epilogue. For frame t the
// product p[z] = sum_j wt[j*Z + z] * x[t*C + j] is formed in registers and then
// combined with y according to Epi:
//   Acc           y += p
//   Store         y  = p
//   AccBiasMix    y  = ((y + p) + bias) + mix * cond[t]
//   StoreBiasMix  y  = (p + bias) + mix * cond[t]
//   BiasResid     y  = resid + (p + bias)          (resid has y's layout)
// Each product is added to y once (the grouping a GEMM uses), so a sum of
// several products rounds like a sum of separate GEMM results. The mix term is
// rounded before it is added (no fused multiply-add across the epilogue).
enum class Epi { Acc, Store, AccBiasMix, StoreBiasMix, BiasResid };

struct EpiArgs {
    const float* bias = nullptr;
    const float* mix = nullptr;
    const float* cond = nullptr;
    const float* resid = nullptr;
};

using KernelFn = void (*)(const float* __restrict wt, const float* __restrict x, float* __restrict y, int n,
                          int Z, int C, const EpiArgs& e);

#if defined(__GNUC__) || defined(__clang__)
#define GPU_NAM_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define GPU_NAM_ALWAYS_INLINE inline
#endif

template <Epi E, int Z>
GPU_NAM_ALWAYS_INLINE void epilogue(float* __restrict yt, const float (&p)[Z], int t, const EpiArgs& e) {
    if constexpr (E == Epi::Acc) {
        for (int z = 0; z < Z; ++z) yt[z] += p[z];
    } else if constexpr (E == Epi::Store) {
        for (int z = 0; z < Z; ++z) yt[z] = p[z];
    } else if constexpr (E == Epi::AccBiasMix || E == Epi::StoreBiasMix) {
        const float* __restrict bias = e.bias;
        const float* __restrict mix = e.mix;
        const float cv = e.cond[t];
        for (int z = 0; z < Z; ++z) {
            float v = E == Epi::AccBiasMix ? yt[z] + p[z] : p[z];
            v = v + bias[z];
            const float m = mix[z] * cv;
            yt[z] = v + m;
        }
    } else {
        const float* __restrict bias = e.bias;
        const float* __restrict rt = e.resid + static_cast<std::ptrdiff_t>(t) * Z;
        for (int z = 0; z < Z; ++z) {
            const float v = p[z] + bias[z];
            yt[z] = rt[z] + v;
        }
    }
}

// Four-lane float vector (GCC/Clang vector extension; lowers to NEON / SSE).
typedef float v4f __attribute__((vector_size(16)));

GPU_NAM_ALWAYS_INLINE v4f load4(const float* p) {
    v4f v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}
GPU_NAM_ALWAYS_INLINE void store4(float* p, v4f v) { __builtin_memcpy(p, &v, sizeof(v)); }

// Vector epilogue for Z % 4 == 0; lane-for-lane the same operations as epilogue().
template <Epi E, int V>
GPU_NAM_ALWAYS_INLINE void epilogue_v(float* __restrict yt, const v4f (&p)[V], int t, const EpiArgs& e) {
    for (int v = 0; v < V; ++v) {
        float* yv = yt + 4 * v;
        if constexpr (E == Epi::Acc) {
            store4(yv, load4(yv) + p[v]);
        } else if constexpr (E == Epi::Store) {
            store4(yv, p[v]);
        } else if constexpr (E == Epi::AccBiasMix || E == Epi::StoreBiasMix) {
            v4f r = E == Epi::AccBiasMix ? load4(yv) + p[v] : p[v];
            r = r + load4(e.bias + 4 * v);
            const v4f m = load4(e.mix + 4 * v) * e.cond[t];
            store4(yv, r + m);
        } else {
            const v4f r = p[v] + load4(e.bias + 4 * v);
            store4(yv, load4(e.resid + static_cast<std::ptrdiff_t>(t) * (4 * V) + 4 * v) + r);
        }
    }
}

template <int Z, int C, Epi E>
void kernel_fixed(const float* __restrict wt, const float* __restrict x, float* __restrict y, int n, int, int,
                  const EpiArgs& e) {
    // Group size keeps the G*Z accumulators inside the 32 NEON registers.
    constexpr int G = Z <= 4 ? 8 : (Z <= 8 ? 6 : (Z <= 16 ? 4 : (Z <= 32 ? 2 : 1)));
    int t = 0;
    if constexpr (Z % 4 == 0) {
        constexpr int V = Z / 4;
        for (; t + G <= n; t += G) {
            v4f acc[G][V] = {};
            for (int j = 0; j < C; ++j) {
                v4f w[V];
                for (int v = 0; v < V; ++v) w[v] = load4(wt + j * Z + 4 * v);
                for (int g = 0; g < G; ++g) {
                    const float xv = x[(t + g) * C + j];
                    for (int v = 0; v < V; ++v) acc[g][v] += w[v] * xv;
                }
            }
            for (int g = 0; g < G; ++g) epilogue_v<E, V>(y + (t + g) * Z, acc[g], t + g, e);
        }
        for (; t < n; ++t) {
            v4f acc[V] = {};
            for (int j = 0; j < C; ++j) {
                const float xv = x[t * C + j];
                for (int v = 0; v < V; ++v) acc[v] += load4(wt + j * Z + 4 * v) * xv;
            }
            epilogue_v<E, V>(y + t * Z, acc, t, e);
        }
    } else {
        for (; t + G <= n; t += G) {
            float acc[G][Z] = {};
            for (int j = 0; j < C; ++j) {
                const float* w = wt + j * Z;
                for (int g = 0; g < G; ++g) {
                    const float xv = x[(t + g) * C + j];
                    for (int z = 0; z < Z; ++z) acc[g][z] += w[z] * xv;
                }
            }
            for (int g = 0; g < G; ++g) epilogue<E, Z>(y + (t + g) * Z, acc[g], t + g, e);
        }
        for (; t < n; ++t) {
            float acc[Z] = {};
            for (int j = 0; j < C; ++j) {
                const float* w = wt + j * Z;
                const float xv = x[t * C + j];
                for (int z = 0; z < Z; ++z) acc[z] += w[z] * xv;
            }
            epilogue<E, Z>(y + t * Z, acc, t, e);
        }
    }
}

// Runtime-shape kernel: output channels in vector chunks (16, then 4, then
// scalar) with the same per-frame sum order as kernel_fixed.
template <Epi E, int V>
GPU_NAM_ALWAYS_INLINE void generic_chunk(const float* __restrict wt, const float* __restrict x, float* __restrict y,
                                         int n, int Z, int C, int z0, const EpiArgs& e) {
    EpiArgs ez = e;
    if (ez.bias) ez.bias += z0;
    if (ez.mix) ez.mix += z0;
    if (ez.resid) ez.resid += z0;
    // Two frames per pass share each weight load; a single frame finishes odd n.
    for (int t = 0; t < n;) {
        const int frames = t + 1 < n ? 2 : 1;
        v4f acc2[2][V] = {};
        const float* xt = x + static_cast<std::ptrdiff_t>(t) * C;
        for (int j = 0; j < C; ++j) {
            const float* w = wt + static_cast<std::ptrdiff_t>(j) * Z + z0;
            const float x0 = xt[j];
            const float x1 = frames == 2 ? xt[C + j] : 0.0f;
            for (int v = 0; v < V; ++v) {
                const v4f wv = load4(w + 4 * v);
                acc2[0][v] += wv * x0;
                acc2[1][v] += wv * x1;
            }
        }
        for (int f = 0; f < frames; ++f, ++t) {
            const v4f (&acc)[V] = acc2[f];
            float* yt = y + static_cast<std::ptrdiff_t>(t) * Z + z0;
            for (int v = 0; v < V; ++v) {
                float* yv = yt + 4 * v;
                if constexpr (E == Epi::Acc) {
                    store4(yv, load4(yv) + acc[v]);
                } else if constexpr (E == Epi::Store) {
                    store4(yv, acc[v]);
                } else if constexpr (E == Epi::AccBiasMix || E == Epi::StoreBiasMix) {
                    v4f r = E == Epi::AccBiasMix ? load4(yv) + acc[v] : acc[v];
                    r = r + load4(ez.bias + 4 * v);
                    const v4f m = load4(ez.mix + 4 * v) * e.cond[t];
                    store4(yv, r + m);
                } else {
                    const v4f r = acc[v] + load4(ez.bias + 4 * v);
                    store4(yv, load4(ez.resid + static_cast<std::ptrdiff_t>(t) * Z + 4 * v) + r);
                }
            }
        }
    }
}

template <Epi E>
void kernel_generic(const float* __restrict wt, const float* __restrict x, float* __restrict y, int n, int Z,
                    int C, const EpiArgs& e) {
    int z0 = 0;
    for (; z0 + 16 <= Z; z0 += 16) generic_chunk<E, 4>(wt, x, y, n, Z, C, z0, e);
    for (; z0 + 4 <= Z; z0 += 4) generic_chunk<E, 1>(wt, x, y, n, Z, C, z0, e);
    for (; z0 < Z; ++z0) {
        for (int t = 0; t < n; ++t) {
            const float* xt = x + static_cast<std::ptrdiff_t>(t) * C;
            float p = 0.0f;
            for (int j = 0; j < C; ++j) p += wt[static_cast<std::ptrdiff_t>(j) * Z + z0] * xt[j];
            float& yz = y[static_cast<std::ptrdiff_t>(t) * Z + z0];
            if constexpr (E == Epi::Acc) {
                yz += p;
            } else if constexpr (E == Epi::Store) {
                yz = p;
            } else if constexpr (E == Epi::AccBiasMix || E == Epi::StoreBiasMix) {
                float v = E == Epi::AccBiasMix ? yz + p : p;
                v = v + e.bias[z0];
                const float m = e.mix[z0] * e.cond[t];
                yz = v + m;
            } else {
                const float v = p + e.bias[z0];
                yz = e.resid[static_cast<std::ptrdiff_t>(t) * Z + z0] + v;
            }
        }
    }
}

// z[r] += sum_j wt[j*R + r] * v[j] for one input vector (a recurrent step):
// rows in register tiles of 32 so the accumulators stay resident across all j.
inline void matvec_acc(const float* __restrict wt, const float* __restrict v, float* __restrict z, int R, int C) {
    constexpr int T = 32;
    int r0 = 0;
    for (; r0 + T <= R; r0 += T) {
        float acc[T];
        for (int i = 0; i < T; ++i) acc[i] = z[r0 + i];
        for (int j = 0; j < C; ++j) {
            const float* w = wt + static_cast<std::ptrdiff_t>(j) * R + r0;
            const float vj = v[j];
            for (int i = 0; i < T; ++i) acc[i] += w[i] * vj;
        }
        for (int i = 0; i < T; ++i) z[r0 + i] = acc[i];
    }
    for (; r0 + 8 <= R; r0 += 8) {
        float acc[8];
        for (int i = 0; i < 8; ++i) acc[i] = z[r0 + i];
        for (int j = 0; j < C; ++j) {
            const float* w = wt + static_cast<std::ptrdiff_t>(j) * R + r0;
            const float vj = v[j];
            for (int i = 0; i < 8; ++i) acc[i] += w[i] * vj;
        }
        for (int i = 0; i < 8; ++i) z[r0 + i] = acc[i];
    }
    for (; r0 < R; ++r0) {
        float acc = z[r0];
        for (int j = 0; j < C; ++j) acc += wt[static_cast<std::ptrdiff_t>(j) * R + r0] * v[j];
        z[r0] = acc;
    }
}

namespace detail {
template <Epi E, int Z, int... Cs>
inline KernelFn pick_c(int C) {
    KernelFn f = nullptr;
    (void)((C == Cs ? (f = &kernel_fixed<Z, Cs, E>, true) : false) || ...);
    return f;
}
template <Epi E, int... Zs>
inline KernelFn pick_any(int Z, int C) {
    KernelFn f = nullptr;
    (void)((Z == Zs ? (f = pick_c<E, Zs, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32>(C), true) : false) || ...);
    return f;
}
// Square (Z == C) and gated (Z == 2C) shapes only: the dilated-conv taps and 1x1s.
template <Epi E, int... Cs>
inline KernelFn pick_layer(int Z, int C) {
    KernelFn f = nullptr;
    (void)(((C == Cs && Z == Cs) ? (f = &kernel_fixed<Cs, Cs, E>, true)
            : (C == Cs && Z == 2 * Cs) ? (f = &kernel_fixed<2 * Cs, Cs, E>, true)
                                       : false) || ...);
    return f;
}
}  // namespace detail

// Specialized kernel for the channel counts real captures use (standard / lite /
// feather / nano presets and their gated doubles); any other shape runs the
// generic loop, which forms the same sums in the same order.
inline KernelFn pick_kernel(Epi e, int Z, int C) {
    KernelFn f = nullptr;
    switch (e) {
        case Epi::Acc: f = detail::pick_any<Epi::Acc, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32>(Z, C); break;
        case Epi::Store: f = detail::pick_any<Epi::Store, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32>(Z, C); break;
        case Epi::AccBiasMix: f = detail::pick_layer<Epi::AccBiasMix, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32>(Z, C); break;
        case Epi::StoreBiasMix: f = detail::pick_any<Epi::StoreBiasMix, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32>(Z, C); break;
        case Epi::BiasResid: f = detail::pick_layer<Epi::BiasResid, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32>(Z, C); break;
    }
    if (f) return f;
    switch (e) {
        case Epi::Acc: return &kernel_generic<Epi::Acc>;
        case Epi::Store: return &kernel_generic<Epi::Store>;
        case Epi::AccBiasMix: return &kernel_generic<Epi::AccBiasMix>;
        case Epi::StoreBiasMix: return &kernel_generic<Epi::StoreBiasMix>;
        case Epi::BiasResid: return &kernel_generic<Epi::BiasResid>;
    }
    return &kernel_generic<Epi::Acc>;
}

}  // namespace kernels
}  // namespace pulp::examples::nam
