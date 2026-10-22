#pragma once

// Block-streaming CPU engine for the NAM WaveNet (A1) architecture.
//
// Computes the same network as the per-sample oracle in nam_model.hpp, but
// layer-major over a run of frames instead of sample-major through every layer:
// each dilated conv tap, input mixin, 1x1 and head is one small matrix product
// over the whole run (nam_kernels.hpp), and each activation is one array-wide
// call. Per-sample virtual-free scalar matvecs become register-blocked products
// whose weights stay in L1 across frames.
//
// Causal history: every layer keeps its own input history as a linear time-major
// buffer of `reach + span` frames. A run is written at the write position and the
// k-th tap reads the contiguous run starting (K-1-k)*dilation frames earlier, so
// no tap ever wraps. When a run would overflow, the newest `reach` frames are
// moved to the front (amortized one frame copy per frame).
//
// Real-time contract: everything is sized in build(); process() never allocates,
// for any n (runs longer than kRun are split). Value-copyable (no self pointers).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "nam_kernels.hpp"
#include "nam_wavenet_config.hpp"

namespace pulp::examples::nam {

class WaveNetBlockEngine {
public:
    static constexpr int kRun = 128;   // frames per inner pass

    // Build from the parsed array configs and the flat .nam weight blob (same
    // consumption order as the oracle, head_scale last). Returns false for shapes
    // the engine does not cover (the caller then keeps the scalar oracle).
    bool build(const std::vector<LayerArrayConfig>& cfgs, const float* w, std::size_t count) {
        arrays_.clear();
        ok_ = false;
        if (cfgs.empty()) return false;
        const float* p = w;
        const float* end = w + count;
        auto take = [&](std::size_t n) -> const float* {
            if (static_cast<std::size_t>(end - p) < n) return nullptr;
            const float* r = p;
            p += n;
            return r;
        };
        std::size_t max_c = 1, max_z = 1, max_h = 1;
        for (std::size_t ai = 0; ai < cfgs.size(); ++ai) {
            const LayerArrayConfig& c = cfgs[ai];
            if (c.condition_size != 1 || c.channels <= 0 || c.kernel_size <= 0 || c.head_size <= 0) return false;
            if (ai > 0 && cfgs[ai - 1].head_size != c.channels) return false;
            Array a;
            a.in = c.input_size;
            a.C = c.channels;
            a.H = c.head_size;
            const int C = a.C;
            const float* rw = take(static_cast<std::size_t>(C) * a.in);
            if (!rw) return false;
            a.rechannel = make_mat(rw, C, a.in, 1, 0, kernels::Epi::Store);
            for (int d : c.dilations) {
                Layer L;
                L.K = c.kernel_size;
                L.dilation = d;
                L.reach = (L.K - 1) * d;
                L.gated = c.gated;
                L.act = parse_activation(c.activation);
                const int Z = c.gated ? 2 * C : C;
                L.Z = Z;
                const float* cw = take(static_cast<std::size_t>(Z) * C * L.K);
                const float* cb = take(static_cast<std::size_t>(Z));
                const float* mw = take(static_cast<std::size_t>(Z));
                const float* lw = take(static_cast<std::size_t>(C) * C);
                const float* lb = take(static_cast<std::size_t>(C));
                if (!cw || !cb || !mw || !lw || !lb) return false;
                // Taps run oldest first: the first stores, the last adds bias + mixin.
                for (int k = 0; k < L.K; ++k) {
                    const bool first = k == 0, last = k == L.K - 1;
                    const kernels::Epi e = first && last ? kernels::Epi::StoreBiasMix
                                           : first       ? kernels::Epi::Store
                                           : last        ? kernels::Epi::AccBiasMix
                                                         : kernels::Epi::Acc;
                    L.taps.push_back(make_mat(cw, Z, C, L.K, k, e));
                }
                L.conv_b.assign(cb, cb + Z);
                L.mixin.assign(mw, mw + Z);
                L.l1 = make_mat(lw, C, C, 1, 0, kernels::Epi::BiasResid);
                L.l1_b.assign(lb, lb + C);
                L.span = std::max(L.reach, 4 * kRun);
                L.hist.assign(static_cast<std::size_t>(L.reach + L.span) * C, 0.0f);
                max_z = std::max<std::size_t>(max_z, static_cast<std::size_t>(Z));
                a.layers.push_back(std::move(L));
            }
            if (a.layers.empty()) return false;
            const float* hw = take(static_cast<std::size_t>(a.H) * C);
            if (!hw) return false;
            a.head = make_mat(hw, a.H, C, 1, 0, kernels::Epi::StoreBiasMix);
            a.head_b.assign(static_cast<std::size_t>(a.H), 0.0f);
            a.head_mix.assign(static_cast<std::size_t>(a.H), 0.0f);   // head has no mixin
            if (c.head_bias) {
                const float* hb = take(static_cast<std::size_t>(a.H));
                if (!hb) return false;
                a.head_b.assign(hb, hb + a.H);
            }
            max_c = std::max<std::size_t>(max_c, static_cast<std::size_t>(C));
            max_h = std::max<std::size_t>(max_h, static_cast<std::size_t>(a.H));
            arrays_.push_back(std::move(a));
        }
        const float* hs = take(1);
        if (!hs || p != end) return false;
        head_scale_ = *hs;
        if (arrays_.front().in != 1) return false;
        for (std::size_t ai = 1; ai < arrays_.size(); ++ai)
            if (arrays_[ai].in != arrays_[ai - 1].C) return false;

        const std::size_t run = static_cast<std::size_t>(kRun);
        z_.assign(run * max_z, 0.0f);
        act_.assign(run * max_c, 0.0f);
        acc_.assign(run * max_c, 0.0f);
        layer_out_[0].assign(run * max_c, 0.0f);
        layer_out_[1].assign(run * max_c, 0.0f);
        head_out_[0].assign(run * max_h, 0.0f);
        head_out_[1].assign(run * max_h, 0.0f);
        ok_ = true;
        reset();
        return true;
    }

    bool ok() const { return ok_; }
    float head_scale() const { return head_scale_; }
    void set_tanh_mode(TanhMode m) { mode_ = m; }
    TanhMode tanh_mode() const { return mode_; }

    void reset() {
        for (Array& a : arrays_)
            for (Layer& L : a.layers) {
                std::fill(L.hist.begin(), L.hist.end(), 0.0f);
                L.wpos = L.reach;
            }
    }

    void process(const float* in, float* out, std::uint32_t n) {
        while (n > 0) {
            const int m = static_cast<int>(std::min<std::uint32_t>(n, static_cast<std::uint32_t>(kRun)));
            run(in, out, m);
            in += m;
            out += m;
            n -= static_cast<std::uint32_t>(m);
        }
    }

private:
    struct Mat {
        std::vector<float> wt;   // transposed: wt[j*Z + z]
        int Z = 0, C = 0;
        kernels::KernelFn fn = nullptr;   // carries this use's epilogue
        void run(const float* x, float* y, int n, const kernels::EpiArgs& e) const {
            fn(wt.data(), x, y, n, Z, C, e);
        }
    };
    struct Layer {
        int K = 1, dilation = 1, reach = 0, span = 0, Z = 1, wpos = 0;
        bool gated = false;
        Activation act = Activation::Tanh;
        std::vector<Mat> taps;
        std::vector<float> conv_b, mixin, l1_b;
        Mat l1;
        std::vector<float> hist;   // (reach + span) frames x C
    };
    struct Array {
        int in = 1, C = 1, H = 1;
        Mat rechannel, head;
        std::vector<float> head_b, head_mix;
        std::vector<Layer> layers;
    };

    // W is [Z][C][K] row-major; extract tap k transposed.
    static Mat make_mat(const float* w, int Z, int C, int K, int k, kernels::Epi epi) {
        Mat m;
        m.Z = Z;
        m.C = C;
        m.wt.resize(static_cast<std::size_t>(Z) * C);
        for (int z = 0; z < Z; ++z)
            for (int j = 0; j < C; ++j)
                m.wt[static_cast<std::size_t>(j) * Z + z] = w[(static_cast<std::size_t>(z) * C + j) * K + k];
        m.fn = kernels::pick_kernel(epi, Z, C);
        return m;
    }

    // Make room for an m-frame run in L's history (keeps the newest `reach`).
    static float* write_slot(Layer& L, int C, int m) {
        if (L.wpos + m > L.reach + L.span) {
            std::memmove(L.hist.data(), L.hist.data() + static_cast<std::size_t>(L.wpos - L.reach) * C,
                         static_cast<std::size_t>(L.reach) * C * sizeof(float));
            L.wpos = L.reach;
        }
        return L.hist.data() + static_cast<std::size_t>(L.wpos) * C;
    }

    void activate(float* d, std::size_t count, Activation a) const {
        switch (a) {
            case Activation::Tanh: kernels::tanh_inplace(d, count, mode_); return;
            case Activation::Fasttanh: kernels::tanh_inplace(d, count, mode_); return;
            case Activation::Sigmoid: kernels::sigmoid_inplace(d, count, TanhMode::Exact); return;
            case Activation::Hardtanh:
                for (std::size_t i = 0; i < count; ++i) d[i] = d[i] < -1.0f ? -1.0f : (d[i] > 1.0f ? 1.0f : d[i]);
                return;
            case Activation::ReLU:
                for (std::size_t i = 0; i < count; ++i) d[i] = d[i] > 0.0f ? d[i] : 0.0f;
                return;
            case Activation::Identity: return;
        }
    }

    void run(const float* x, float* out, int m) {
        const float* layer_in = x;   // array 0 reads the raw input (in == 1)
        const float* head_in = nullptr;
        int cur = 0;
        const kernels::EpiArgs none{};
        for (Array& a : arrays_) {
            const int C = a.C;
            const std::size_t mc = static_cast<std::size_t>(m) * C;
            // Array input -> layer 0's history.
            a.rechannel.run(layer_in, write_slot(a.layers.front(), C, m), m, none);
            // Skip accumulator seeded with the previous array's head output.
            if (head_in) std::copy(head_in, head_in + mc, acc_.begin());
            else std::fill(acc_.begin(), acc_.begin() + static_cast<std::ptrdiff_t>(mc), 0.0f);

            float* lout = layer_out_[cur].data();
            for (std::size_t li = 0; li < a.layers.size(); ++li) {
                Layer& L = a.layers[li];
                const int Z = L.Z;
                const float* hcur = L.hist.data() + static_cast<std::size_t>(L.wpos) * C;
                // z = (sum_k W_k * history[t - (K-1-k)*d] + bias) + mixin * condition
                float* z = z_.data();
                const kernels::EpiArgs conv_e{L.conv_b.data(), L.mixin.data(), x, nullptr};
                for (int k = 0; k < L.K; ++k) {
                    const int back = (L.K - 1 - k) * L.dilation;
                    L.taps[static_cast<std::size_t>(k)].run(hcur - static_cast<std::ptrdiff_t>(back) * C, z, m, conv_e);
                }
                // Activation -> act (m x C).
                float* act = act_.data();
                if (!L.gated) {
                    activate(z, static_cast<std::size_t>(m) * Z, L.act);
                    act = z;
                } else {
                    for (int t = 0; t < m; ++t) {
                        float* zt = z + static_cast<std::size_t>(t) * Z;
                        activate(zt, static_cast<std::size_t>(C), L.act);
                        kernels::sigmoid_inplace(zt + C, static_cast<std::size_t>(C), TanhMode::Exact);
                        float* at = act + static_cast<std::size_t>(t) * C;
                        for (int c = 0; c < C; ++c) at[c] = zt[c] * zt[C + c];
                    }
                }
                // Skip sum.
                for (std::size_t i = 0; i < mc; ++i) acc_[i] += act[i];
                // Residual: next = input + (W1 * act + b1), written straight into the
                // next layer's history (or this array's output after the last layer).
                // write_slot only ever moves the next layer's buffer, so hcur stays valid.
                float* next = (li + 1 < a.layers.size()) ? write_slot(a.layers[li + 1], C, m) : lout;
                L.l1.run(act, next, m, kernels::EpiArgs{L.l1_b.data(), nullptr, nullptr, hcur});
                L.wpos += m;
            }
            // Head: head_out = Wh * acc + head_b.
            float* hout = head_out_[cur].data();
            a.head.run(acc_.data(), hout, m, kernels::EpiArgs{a.head_b.data(), a.head_mix.data(), x, nullptr});
            layer_in = lout;
            head_in = hout;
            cur ^= 1;
        }
        const int H = arrays_.back().H;
        for (int t = 0; t < m; ++t) out[t] = head_scale_ * head_in[static_cast<std::size_t>(t) * H];
    }

    std::vector<Array> arrays_;
    float head_scale_ = 1.0f;
    TanhMode mode_ = TanhMode::Exact;
    bool ok_ = false;
    std::vector<float> z_, act_, acc_;
    std::vector<float> layer_out_[2], head_out_[2];
};

}  // namespace pulp::examples::nam
