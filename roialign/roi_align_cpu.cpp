#include <ATen/ATen.h>

using std::vector;

template<typename T>
struct PreCalc {
    // left_top, right_top, left_bottom, right_bottom
    int pos1, pos2, pos3, pos4;
    T w1, w2, w3, w4;
};

template<class T>
inline void add(const T &val, T *address) {
    *address += val;
}

/* -----------------------------begin for forward---------------------------------  */
template<typename T>
void pre_calc_for_bilinear(const int h, const int w, const int pool_h, const int pool_w, int b_grid_h, int b_grid_w,
                           T start_y, T start_x, T b_size_h, T b_size_w, vector<PreCalc<T>> &pre_calc) {
    int idx = 0;
    for (int ph = 0; ph < pool_h; ++ph) {
        for (int pw = 0; pw < pool_w; ++pw) {
            for (int iy = 0; iy < b_grid_h; ++iy) {
                const T yy = start_y + ph * b_size_h + static_cast<T>(iy + 0.5f) * b_size_h / static_cast<T>(b_grid_h);
                for (int ix = 0; ix < b_grid_w; ++ix) {
                    const T xx =
                            start_x + pw * b_size_w + static_cast<T>(ix + 0.5f) * b_size_w / static_cast<T>(b_grid_w);
                    T x = xx, y = yy;
                    // situation 1: out of range
                    if (y < -1.0 || y > h || x < -1.0 || x > w) {
                        PreCalc<T> pc{0, 0, 0, 0, 0, 0, 0, 0};
                        pre_calc[idx] = pc;
                        idx += 1;
                        continue;
                    }
                    // not exceed 1.0
                    y = y <= 0 ? 0 : (y >= h - 1 ? h - 1 : y);
                    x = x <= 0 ? 0 : (x >= w - 1 ? w - 1 : x);
                    int y_low = (int) y;
                    int x_low = (int) x;
                    int y_high = y_low >= h - 1 ? y_low : y_low + 1;
                    int x_high = x_low >= w - 1 ? x_low : x_low + 1;
                    T ly = y - y_low, lx = x - x_low;
                    T hy = 1.0 - ly, hx = 1.0 - lx;
                    T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
                    // in the feature map's position and correspond weights
                    PreCalc<T> pc;
                    pc.pos1 = y_low * w + x_low;
                    pc.pos2 = y_low * w + x_high;
                    pc.pos3 = y_high * w + x_low;
                    pc.pos4 = y_high * w + x_high;
                    pc.w1 = w1, pc.w2 = w2, pc.w3 = w3, pc.w4 = w4;
                    pre_calc[idx] = pc;
                    idx += 1;
                } // b_grid_w
            } // b_grid_h
        } // pool_w
    } // pool_h
}


template<typename T>
void roi_align_forward(const T *feat, const T *rois, const vector<int64_t> &feat_size,
                       const vector<int64_t> &rois_size, const T &scale, const int ratio, T *out) {
    const int n_rois = rois_size[0], col_rois = rois_size[1], pool_h = rois_size[2], pool_w = rois_size[3];
    const int channel = feat_size[1], h = feat_size[2], w = feat_size[3];
    // #pragma omp parallel for
    for (int n = 0; n < n_rois; ++n) {
        int idx_n = n * channel * pool_h * pool_w;
        // rois data
        const T *offset_rois = rois + col_rois * n;
        int roi_batch_idx = 0;
        if (col_rois == 5) {
            roi_batch_idx = offset_rois[0];
            ++offset_rois;
        }
        // Do not using rounding; this implementation detail is critical
        T start_x = offset_rois[0] * scale;
        T start_y = offset_rois[1] * scale;
        T end_x = offset_rois[2] * scale;
        T end_y = offset_rois[3] * scale;

        // Force malformed ROIs to be 1x1
        T roi_w = std::max(end_x - start_x, (T) 1.);
        T roi_h = std::max(end_y - start_y, (T) 1.);
        T bin_size_w = roi_w / static_cast<T>(pool_w);
        T bin_size_h = roi_h / static_cast<T>(pool_h);

        // We use roi_bin_grid to sample the grid and mimic integral
        int bin_grid_h = (ratio > 0) ? ratio : std::ceil(roi_h / pool_h);
        int bin_grid_w = (ratio > 0) ? ratio : std::ceil(roi_w / pool_w);
        // We do average (integral) pooling inside a bin
        const T count = bin_grid_h * bin_grid_w;
        // get each bin's corresponding position and weights
        std::vector<PreCalc<T>> pre_calc(count * pool_h * pool_w);
        pre_calc_for_bilinear(h, w, pool_h, pool_w, bin_grid_h, bin_grid_w, start_y, start_x, bin_size_h, bin_size_w,
                              pre_calc);
        // map to feature map
        for (int c = 0; c < channel; ++c) {
            int idx_nc = idx_n + c * pool_w * pool_h;
            const T *offset_feat = feat + (roi_batch_idx * channel + c) * h * w;
            int pre_calc_idx = 0;
            for (int ph = 0; ph < pool_h; ++ph) {
                for (int pw = 0; pw < pool_w; ++pw) {
                    int idx = idx_nc + ph * pool_w + pw;
                    T output_val = 0.;
                    for (int iy = 0; iy < bin_grid_h; ++iy) {
                        for (int ix = 0; ix < bin_grid_w; ++ix) {
                            PreCalc<T> pc = pre_calc[pre_calc_idx];
                            output_val += pc.w1 * offset_feat[pc.pos1] + pc.w2 * offset_feat[pc.pos2] +
                                          pc.w3 * offset_feat[pc.pos3] + pc.w4 * offset_feat[pc.pos4];
                            pre_calc_idx += 1;
                        }
                    }
                    output_val /= count;
                    out[idx] = output_val;
                }  // for pw
            } // for ph
        } // for c
    }  // for rois_n
}


// input: BxCxHxW;  rois: Kx5
at::Tensor roi_align_forward_cpu(const at::Tensor &feat, const at::Tensor &rois, int64_t pool_h, int64_t pool_w,
                                 double scale, int64_t sample) {
    AT_CHECK(feat.ndimension() == 4, "Feature should be BxCxHxW forms");
    AT_CHECK(feat.is_contiguous(), "Feature should be contiguous");
    AT_CHECK(rois.ndimension() == 2, "ROI Proposals should be Kx5 forms");
    AT_CHECK(rois.size(1) == 5, "ROI proposals should be Kx5 forms");
    AT_CHECK(rois.is_contiguous(), "ROI proposals should be contiguous.");

    const vector<int64_t> rois_size = {rois.size(0), rois.size(1), pool_h, pool_w};
    const vector<int64_t> feat_size = {feat.size(0), feat.size(1), feat.size(2), feat.size(3)};

    auto output = feat.type().tensor({rois_size[0], feat_size[1], pool_h, pool_w});
    roi_align_forward(feat.data<float>(), rois.data<float>(), feat_size, rois_size, static_cast<float>(scale), sample,
                      output.data<float>());
    return output;
}
/*------------------------------end of forward-----------------------------*/

/*------------------------------begin for backward-----------------------------*/
template<typename T>
void bilinear_interpolate_gradient(const int h, const int w, T y, T x, PreCalc<T> &pc) {
    if (y < -1.0 || y > h || x < -1.0 || x > w) {
        pc = {-1, -1, -1, -1, 0., 0., 0., 0.};
        return;
    }
    // not exceed 1.0
    y = y <= 0 ? 0 : (y >= h - 1 ? h - 1 : y);
    x = x <= 0 ? 0 : (x >= w - 1 ? w - 1 : x);
    int y_low = (int) y;
    int x_low = (int) x;
    int y_high = y_low >= h - 1 ? y_low : y_low + 1;
    int x_high = x_low >= w - 1 ? x_low : x_low + 1;
    pc.pos1 = y_low * w + x_low;
    pc.pos2 = y_low * w + x_high;
    pc.pos3 = y_high * w + x_low;
    pc.pos4 = y_high * w + x_high;
    T ly = y - y_low, lx = x - x_low;
    T hy = 1.0 - ly, hx = 1.0 - lx;
    pc.w1 = hy * hx, pc.w2 = hy * lx, pc.w3 = ly * hx, pc.w4 = ly * lx;
}


template<typename T>
void roi_align_backward(int total, const T *rois, T *grad_out, const T &scale, const vector<int64_t> feat_size,
                        const int pool_h, const int pool_w, const int rois_col, const int sample, T *grad_in) {
    // total=nxcxphxpw
    auto channel = feat_size[0], h = feat_size[1], w = feat_size[2];
    for (int idx = 0; idx < total; ++idx) {
        int pw = idx % pool_w;
        int ph = (idx / pool_w) % pool_h;
        int c = (idx / pool_h / pool_w) % channel;
        int n = idx / pool_h / pool_w / channel;

        const T *offset_rois = rois + n * rois_col;
        int roi_batch_idx = 0;
        if (rois_col == 5) {
            roi_batch_idx = offset_rois[0];
            ++offset_rois;
        }
        // Do not using rounding; this implementation detail is critical
        T start_x = offset_rois[0] * scale;
        T start_y = offset_rois[1] * scale;
        T end_x = offset_rois[2] * scale;
        T end_y = offset_rois[3] * scale;

        // Force malformed ROIs to be 1x1
        T roi_w = std::max(end_x - start_x, (T) 1.0);
        T roi_h = std::max(end_y - start_y, (T) 1.0);
        T b_size_h = roi_h / static_cast<T>(pool_h);
        T b_size_w = roi_w / static_cast<T>(pool_w);

        T *offset_grad_in = grad_in + (roi_batch_idx * channel + c) * h * w;
        T *offset_grad_out = grad_out + (n * channel + c) * pool_h * pool_w;
        T grad_out_this_bin = offset_grad_out[ph * pool_w + pw];

        // We use roi_bin_grid to sample the grid and mimic integral
        int roi_bin_grid_h = (sample > 0) ? sample : std::ceil(roi_h / pool_h);
        int roi_bin_grid_w = (sample > 0) ? sample : std::ceil(roi_w / pool_w);
        // We do average (integral) pooling inside a bin
        const int count = roi_bin_grid_h * roi_bin_grid_w;
        PreCalc<T> pc;
        for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            const T y = start_y + ph * b_size_h +
                        static_cast<T>(iy + .5f) * b_size_h / static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                const T x = start_x + pw * b_size_w +
                            static_cast<T>(ix + .5f) * b_size_w / static_cast<T>(roi_bin_grid_w);
                bilinear_interpolate_gradient(h, w, y, x, pc);
                T g1 = grad_out_this_bin * pc.w1 / count;
                T g2 = grad_out_this_bin * pc.w2 / count;
                T g3 = grad_out_this_bin * pc.w3 / count;
                T g4 = grad_out_this_bin * pc.w4 / count;
                // update grad_out
                if (pc.pos1 >= 0 && pc.pos2 >= 0 && pc.pos3 >= 0 && pc.pos4 >= 0) {
                    add(g1, offset_grad_in + pc.pos1);
                    add(g2, offset_grad_in + pc.pos2);
                    add(g3, offset_grad_in + pc.pos3);
                    add(g4, offset_grad_in + pc.pos4);
                }
            }  // for ix
        }  // for iy
    }  // for
}


at::Tensor
roi_align_backward_cpu(const at::Tensor &rois, const at::Tensor &grad_out, int64_t b_size, int64_t channel,
                       int64_t h, int64_t w, int64_t pool_h, int64_t pool_w, double scale, int64_t sample) {
    AT_CHECK(rois.ndimension() == 2 && rois.size(1) == 5, "ROI Proposals should be Kx5 forms")
    AT_CHECK(rois.is_contiguous(), "ROI proposals should be contiguous.")
    auto rois_col = rois.size(1);
    auto grad_in = rois.type().tensor({b_size, channel, h, w});
    grad_in.zero_();
    std::cout << grad_in << std::endl;
    roi_align_backward(grad_out.numel(), rois.data<float>(), grad_out.data<float>(), static_cast<float>(scale),
                       {channel, h, w}, pool_h, pool_w, rois_col, sample, grad_in.data<float>());
    return grad_in;
}
