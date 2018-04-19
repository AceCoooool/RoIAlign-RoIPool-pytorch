from torch.nn import Module
from torch.autograd import Function
import roi_pool_cpu
import roi_pool_cuda


class ROIPoolFunction(Function):
    @staticmethod
    def forward(ctx, feat, rois, pool_h, pool_w, scale, train):
        ctx.rois = rois
        ctx.feat_size = feat.size()
        ctx.pool_h = pool_h
        ctx.pool_w = pool_w
        if train:
            ctx.memory = torch.zeros((rois.size(0), feat.size(1), pool_h, pool_w), dtype=torch.int)
        else:
            ctx.memory = torch.zeros(0)
        if feat.is_cuda:
            ctx.memory = ctx.memory.cuda()
            output = roi_pool_cuda.forward_cuda(feat, rois, pool_h, pool_w, scale, ctx.memory)
        else:
            output = roi_pool_cpu.forward_cpu(feat, rois, pool_h, pool_w, scale, ctx.memory)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        rois = ctx.rois
        feat_size = ctx.feat_size
        pool_h = ctx.pool_h
        pool_w = ctx.pool_w
        memory = ctx.memory
        grad_out = grad_out.contiguous() if not grad_out.is_contiguous() else grad_out
        if grad_out.is_cuda:
            grad_in = roi_pool_cuda.backward_cuda(rois, grad_out, feat_size[0], feat_size[1], feat_size[2],
                                                  feat_size[3], pool_h, pool_w, memory)
        else:
            grad_in = roi_pool_cpu.backward_cpu(rois, grad_out, feat_size[0], feat_size[1], feat_size[2],
                                                feat_size[3], pool_h, pool_w, memory)
        # Note: the backward return number is corresponding to the ctx variable
        return grad_in, None, None, None, None, None


class ROIPool(Module):
    def __init__(self, pool_h, pool_w, scale):
        super(ROIPool, self).__init__()
        self.pool_h, self.pool_w = int(pool_h), int(pool_w)
        self.scale = float(scale)

    # feat: BxCxHxW,  rois: Kx5 (batch_idx, xmin, ymin, xmax, ymax) without normalize
    def forward(self, feat, rois):
        output = ROIPoolFunction.apply(feat, rois, self.pool_h, self.pool_w, self.scale, self.training)
        return output


if __name__ == '__main__':
    import torch

    print('------------test on cpu------------')
    roi_pool = ROIPool(2, 2, 0.5)
    feat = torch.arange(64).view(1, 1, 8, 8)
    # Note: first element is batch_idx
    rois = torch.Tensor([0, 1.6, 1.6, 9.2, 11.0]).view(-1, 5)
    feat.requires_grad = True
    out = roi_pool(feat, rois)
    print(out)
    out.sum().backward()
    print(feat.grad)

    if torch.cuda.is_available():
        print('------------test on gpu------------')
        feat = feat.detach().cuda()
        rois = rois.cuda()
        feat.requires_grad = True
        out = roi_pool(feat, rois)
        print(out)
        temp = out.sum()
        temp.backward()
        print(feat.grad)
    else:
        print('You device have not a GPU')
