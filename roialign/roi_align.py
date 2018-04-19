from torch.nn import Module
from torch.autograd import Function
import roi_align_cpu
import roi_align_cuda


class ROIAlignFunction(Function):
    @staticmethod
    def forward(ctx, feat, rois, pool_h, pool_w, scale, sampling):
        ctx.rois = rois
        ctx.feat_size = feat.size()
        ctx.pool_h = pool_h
        ctx.pool_w = pool_w
        ctx.scale = scale
        ctx.sampling = sampling  # sampling number in bin
        if feat.is_cuda:
            output = roi_align_cuda.forward_cuda(feat, rois, pool_h, pool_w, scale, sampling)
        else:
            output = roi_align_cpu.forward_cpu(feat, rois, pool_h, pool_w, scale, sampling)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        rois = ctx.rois
        feat_size = ctx.feat_size
        pool_h = ctx.pool_h
        pool_w = ctx.pool_w
        scale = ctx.scale
        sampling = ctx.sampling
        grad_out = grad_out.contiguous() if not grad_out.is_contiguous() else grad_out
        if grad_out.is_cuda:
            grad_in = roi_align_cuda.backward_cuda(rois, grad_out, feat_size[0], feat_size[1], feat_size[2],
                                                  feat_size[3], pool_h, pool_w, scale, sampling)
        else:
            grad_in = roi_align_cpu.backward_cpu(rois, grad_out, feat_size[0], feat_size[1], feat_size[2], feat_size[3],
                                                 pool_h, pool_w, scale, sampling)
        # Note: the backward return number is corresponding to the forward parameters number
        return grad_in, None, None, None, None, None


class ROIAlign(Module):
    def __init__(self, pool_h, pool_w, scale, sampling=0):
        super(ROIAlign, self).__init__()
        self.pool_h, self.pool_w = int(pool_h), int(pool_w)
        self.scale = float(scale)
        self.sampling = int(sampling)

    # feat: BxCxHxW,  rois: Kx5 (batch_idx, xmin, ymin, xmax, ymax) without normalize
    def forward(self, feat, rois):
        output = ROIAlignFunction.apply(feat, rois, self.pool_h, self.pool_w, self.scale, self.sampling)
        return output


if __name__ == '__main__':
    import torch

    print('------------test on cpu------------')
    roi_align = ROIAlign(2, 2, 0.5, 1)
    feat = torch.arange(64).view(1, 1, 8, 8)
    # Note: first element is batch_idx
    rois = torch.Tensor([0, 1.6, 1.6, 9.2, 11.0]).view(-1, 5)
    feat.requires_grad = True
    out = roi_align(feat, rois)
    print(out)
    out.sum().backward()
    print(feat.grad)

    if torch.cuda.is_available():
        print('------------test on gpu------------')
        feat = feat.detach().cuda()
        rois = rois.cuda()
        feat.requires_grad = True
        out = roi_align(feat, rois)
        print(out)
        temp = out.sum()
        temp.backward()
        print(feat.grad)
    else:
        print('You device have not a GPU')
