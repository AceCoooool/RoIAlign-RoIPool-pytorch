#ifndef ROI_TEMP_H
#define ROI_TEMP_H

#include <ATen/ATen.h>

at::Tensor roi_pool_forward_cpu(const at::Tensor &input, const at::Tensor &rois, int64_t pool_h, int64_t pool_w,
                                double scale, at::Tensor &memory);

at::Tensor roi_pool_backward_cpu(const at::Tensor &rois, const at::Tensor &grad_out, int64_t b_size, int64_t channel,
                                 int64_t h, int64_t w, int64_t pool_h, int64_t pool_w, at::Tensor &memory);

#endif //ROI_TEMP_H
