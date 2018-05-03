#include <torch/torch.h>

// CUDA forward declarations
at::Tensor roi_pool_forward_cuda(const at::Tensor &input, const at::Tensor &rois, int64_t pool_h, int64_t pool_w,
                                 double scale, at::Tensor &memory);

at::Tensor roi_pool_backward_cuda(const at::Tensor &rois, const at::Tensor &grad_out, int64_t b_size, int64_t channel,
                                  int64_t h, int64_t w, int64_t pool_h, int64_t pool_w, const at::Tensor &memory);


// C++ interface
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor roi_pool_forward(const at::Tensor &input, const at::Tensor &rois, int64_t pool_h, int64_t pool_w,
                            double scale, at::Tensor &memory) {
    CHECK_INPUT(input);
    CHECK_INPUT(rois);
    CHECK_INPUT(memory);
    return roi_pool_forward_cuda(input, rois, pool_h, pool_w, scale, memory);
}

at::Tensor roi_pool_backward(const at::Tensor &rois, const at::Tensor &grad_out, int64_t b_size, int64_t channel,
                             int64_t h, int64_t w, int64_t pool_h, int64_t pool_w, const at::Tensor &memory) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(rois);
    CHECK_INPUT(memory);
    return roi_pool_backward_cuda(rois, grad_out, b_size, channel, h, w, pool_h, pool_w, memory);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_cuda", &roi_pool_forward, "roi_pool_forward_cuda");
    m.def("backward_cuda", &roi_pool_backward, "roi_pool_backward_cuda");
}