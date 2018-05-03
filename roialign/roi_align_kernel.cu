#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>


template<typename T>
__device__ __forceinline__

T fmin(T a, T b) {
    return a > b ? b : a;
}

template<typename T>
__device__ __forceinline__

T fmax(T a, T b) {
    return a < b ? b : a;
}

template<typename T>
__device__ __forceinline__

T gpu_atomic_add(const T val, T *address) {
    return atomicAdd(address, val);
}

/* ------------------------------begin of the forward--------------------------- */
template<typename T>
__device__ T

bilinear_interpolate(const T *input, const int h, const int w, T y, T x) {
    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > h || x < -1.0 || x > w) {
        return 0;
    }

    y = y <= 0 ? 0 : (y >= h - 1 ? h - 1 : y);
    x = x <= 0 ? 0 : (x >= w - 1 ? w - 1 : x);

    int y_low = (int) y;
    int x_low = (int) x;
    int y_high = y_low >= h - 1 ? y_low : y_low + 1;
    int x_high = x_low >= w - 1 ? x_low : x_low + 1;

    T ly = y - y_low;
    T lx = x - x_low;
    T hy = 1. - ly, hx = 1. - lx;
    // do bilinear interpolation
    T v1 = input[y_low * w + x_low];
    T v2 = input[y_low * w + x_high];
    T v3 = input[y_high * w + x_low];
    T v4 = input[y_high * w + x_high];
    T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}


template<typename T>
__global__ void
roi_align_forward_kernel(const int total, const T *input, const T *rois, const T scale, const int channel, const int h,
                         const int w, const int pool_h, const int pool_w, const int sampling, T *output) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int pw = idx % pool_w;
        int ph = (idx / pool_w) % pool_h;
        int c = (idx / pool_h / pool_w) % channel;
        int n = idx / pool_h / pool_w / channel;

        const T *offset_rois = rois + n * 5;
        int roi_batch_idx = offset_rois[0];

        // Do not using rounding; this implementation detail is critical
        T start_x = offset_rois[1] * scale;
        T start_y = offset_rois[2] * scale;
        T end_x = offset_rois[3] * scale;
        T end_y = offset_rois[4] * scale;

        // Force malformed ROIs to be 1x1
        T roi_w = fmax(end_x - start_x, (T) 1.);
        T roi_h = fmax(end_y - start_y, (T) 1.);
        T bin_size_h = roi_h / static_cast<T>(pool_h);
        T bin_size_w = roi_w / static_cast<T>(pool_w);

        const T *offset_input = input + (roi_batch_idx * channel * c) * h * w;

        // We use roi_bin_grid to sample the grid and mimic integral
        int bin_grid_h = sampling > 0 ? sampling : ceilf(roi_h / pool_h);
        int bin_grid_w = sampling > 0 ? sampling : ceilf(roi_w / pool_w);
        // We do average (integral) pooling inside a bin
        const T count = bin_grid_h * bin_grid_w;

        T output_val = 0.;
        for (int iy = 0; iy < bin_grid_h; ++iy) {
            T y = start_y + ph * bin_size_h + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(bin_grid_h);
            for (int ix = 0; ix < bin_grid_w; ++ix) {
                T x = start_x + pw * bin_size_w + static_cast<T>(ix + .5f) * bin_size_w /
                                                  static_cast<T>(bin_grid_w);
                T val = bilinear_interpolate(offset_input, h, w, y, x);
                output_val += val;
            }
        }
        output[idx] = output_val /= count;
    }
}

at::Tensor roi_align_forward_cuda(const at::Tensor &input, const at::Tensor &rois, int64_t pool_h, int64_t pool_w,
                                  double scale, int64_t sampling) {
    AT_CHECK(input.ndimension() == 4 && input.is_contiguous(), "Input features should be BxCxHxW and contiguous");
    AT_CHECK(rois.ndimension() == 2 && rois.size(1) == 5, "ROIs should be Kx5 forms");
    AT_CHECK(rois.is_contiguous(), "ROIs should be contiguous");

    auto rois_num = rois.size(0);
    auto channel = input.size(1), h = input.size(2), w = input.size(3);

    auto output = input.type().tensor({rois_num, channel, pool_h, pool_w});

    int64_t total = output.numel();
    const int threads = 1024;
    const int64_t blocks = (total + threads - 1) / threads > 65535 ? 65535 : (total + threads - 1) / threads;

    roi_align_forward_kernel << < blocks, threads >> > (output.numel(), input.data<float>(), rois.data<float>(),
            static_cast<float>(scale), channel, h, w, pool_h, pool_w, sampling, output.data<float>());

    AT_CHECK(cudaGetLastError() == cudaSuccess, "roi_align_forward_kernel failed");
    return output;
}
/* ------------------------------end of the forward--------------------------- */

/* ------------------------------begin of the backward--------------------------- */
template<typename T>
__device__ void bilinear_interpolate_gradient(const int h, const int w, T y, T x, T &w1, T &w2, T &w3, T &w4,
                                              int &pos1, int &pos2, int &pos3, int &pos4) {
    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > h || x < -1.0 || x > w) {
        w1 = w2 = w3 = w4 = 0.;
        pos1 = pos2 = pos3 = pos4 = -1;
        return;
    }

    y = y <= 0 ? 0 : (y >= h - 1 ? h - 1 : y);
    x = x <= 0 ? 0 : (x >= w - 1 ? w - 1 : x);

    int y_low = (int) y;
    int x_low = (int) x;
    int y_high = y_low >= h - 1 ? y_low : y_low + 1;
    int x_high = x_low >= w - 1 ? x_low : x_low + 1;

    pos1 = y_low * w + x_low;
    pos2 = y_low * w + x_high;
    pos3 = y_high * w + x_low;
    pos4 = y_high * w + x_high;

    T ly = y - y_low;
    T lx = x - x_low;
    T hy = 1. - ly, hx = 1. - lx;

    w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
}


template<typename T>
__global__ void roi_align_backward_kernel(const int total, const T *grad_out, const int rois_num,
                                          const T scale, const int channels, const int h, const int w,
                                          const int pool_h, const int pool_w, const int sampling, T *grad_in,
                                          const T *rois, int rois_col) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        // (n, c, ph, pw) is an element in the pooled output
        int pw = idx % pool_w;
        int ph = (idx / pool_w) % pool_h;
        int c = (idx / pool_w / pool_h) % channels;
        int n = idx / pool_w / pool_h / channels;

        const T *offset_rois = rois + n * 5;
        int roi_batch_idx = offset_rois[0];

        // Do not using rounding; this implementation detail is critical
        T start_x = offset_rois[1] * scale;
        T start_y = offset_rois[2] * scale;
        T end_x = offset_rois[3] * scale;
        T end_y = offset_rois[4] * scale;


        // Force malformed ROIs to be 1x1
        T roi_w = fmax(end_x - start_x, (T) 1.);
        T roi_h = fmax(end_y - start_y, (T) 1.);
        T bin_size_h = roi_h / static_cast<T>(pool_h);
        T bin_size_w = roi_w / static_cast<T>(pool_w);

        T *offset_grad_in = grad_in + (roi_batch_idx * channels + c) * h * w;

        const T *offset_grad_out = grad_out + (n * channels + c) * pool_h * pool_w;
        const T grad_out_this_bin = offset_grad_out[ph * pool_w + pw];

        // We use roi_bin_grid to sample the grid and mimic integral
        int roi_bin_grid_h = (sampling > 0) ? sampling : ceilf(roi_h / pool_h); // e.g., = 2
        int roi_bin_grid_w = (sampling > 0) ? sampling : ceilf(roi_w / pool_w);

        // We do average (integral) pooling inside a bin
        const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4
        // PreCalc data type
        T w1, w2, w3, w4;
        int pos1, pos2, pos3, pos4;
        for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
        {
            const T y = start_y + ph * bin_size_h + static_cast<T>(iy + .5f) * bin_size_h /
                                                    static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                const T x = start_x + pw * bin_size_w + static_cast<T>(ix + .5f) * bin_size_w /
                                                        static_cast<T>(roi_bin_grid_w);
                bilinear_interpolate_gradient(h, w, y, x, w1, w2, w3, w4, pos1, pos2, pos3, pos4);

                T g1 = grad_out_this_bin * w1 / count;
                T g2 = grad_out_this_bin * w2 / count;
                T g3 = grad_out_this_bin * w3 / count;
                T g4 = grad_out_this_bin * w4 / count;

                if (pos1 >= 0 && pos2 >= 0 && pos3 >= 0 && pos4 >= 0) {
                    gpu_atomic_add(static_cast<T>(g1), offset_grad_in + pos1);
                    gpu_atomic_add(static_cast<T>(g2), offset_grad_in + pos2);
                    gpu_atomic_add(static_cast<T>(g3), offset_grad_in + pos3);
                    gpu_atomic_add(static_cast<T>(g4), offset_grad_in + pos4);
                } // if
            } // ix
        } // iy
    } // CUDA_1D_KERNEL_LOOP
} // RoIAlignBackward

at::Tensor roi_align_backward_cuda(const at::Tensor &rois, const at::Tensor &grad_out, int64_t b_size, int64_t channel,
                                   int64_t h, int64_t w, int64_t pool_h, int64_t pool_w, double scale,
                                   int64_t sampling) {
    AT_CHECK(rois.ndimension() == 2 && rois.size(1) == 5, "ROIs should be Kx5 forms");
    AT_CHECK(rois.is_contiguous(), "ROIs should be contiguous");

    auto rois_num = rois.size(0), rois_col = rois.size(1);

    auto grad_in = rois.type().tensor({b_size, channel, h, w});
    grad_in.zero_();

    int64_t total = grad_out.numel();
    const int threads = 1024;
    const int64_t blocks = (total + threads - 1) / threads > 65535 ? 65535 : (total + threads - 1) / threads;

    roi_align_backward_kernel << < blocks, threads >> > (grad_out.numel(), grad_out.data<float>(), rois_num,
            static_cast<float>(scale), channel, h, w, pool_h, pool_w, sampling, grad_in.data<float>(),
            rois.data<float>(), rois_col);

    AT_CHECK(cudaGetLastError() == cudaSuccess, "roi_align_forward_kernel failed");
    return grad_in;
}