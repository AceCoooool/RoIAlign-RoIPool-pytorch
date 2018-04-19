#include <torch/torch.h>

// CUDA forward declarations
at::Tensor roi_align_forward_cuda(const at::Tensor &input, const at::Tensor &rois, int64_t pool_h, int64_t pool_w,
                                  double scale, int64_t sampling);

at::Tensor roi_align_backward_cuda(const at::Tensor &rois, const at::Tensor &grad_out, int64_t b_size, int64_t channel,
                                   int64_t h, int64_t w, int64_t pool_h, int64_t pool_w, double scale,
                                   int64_t sampling);


// C++ interface
#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor roi_align_forward(const at::Tensor &input, const at::Tensor &rois, int64_t pool_h, int64_t pool_w,
                             double scale, int64_t sampling) {
    CHECK_INPUT(input);
    CHECK_INPUT(rois);
    return roi_align_forward_cuda(input, rois, pool_h, pool_w, scale, sampling);
}

at::Tensor roi_align_backward(const at::Tensor &rois, const at::Tensor &grad_out, int64_t b_size, int64_t channel,
                              int64_t h, int64_t w, int64_t pool_h, int64_t pool_w, double scale, int64_t sampling) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(rois);
    return roi_align_backward_cuda(rois, grad_out, b_size, channel, h, w, pool_h, pool_w, scale, sampling);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_cuda", &roi_align_forward, "roi_align_forward_cuda");
    m.def("backward_cuda", &roi_align_backward, "roi_align_backward_cuda");
}