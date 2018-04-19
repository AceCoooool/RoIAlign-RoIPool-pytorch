#include <torch/torch.h>
#include "roi_align_cpu.cpp"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_cpu", &roi_align_forward_cpu, "roi_align_forward_cpu");
    m.def("backward_cpu", &roi_align_backward_cpu, "roi_align_backward_cpu");
}