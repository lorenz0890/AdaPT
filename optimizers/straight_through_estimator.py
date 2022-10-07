import torch
from torch import nn
import torch.nn.functional as F


class STETanhFunction(torch.autograd.Function):
    """
    https://www.hassanaskary.com/python/pytorch/deep%20learning/2020/09/19/intuitive-explanation-of-straight-through-estimators.html
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        #return (input > 0).float()
        return torch.tanh(input)

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class STEReLUFunction(torch.autograd.Function):
    """
    https://www.hassanaskary.com/python/pytorch/deep%20learning/2020/09/19/intuitive-explanation-of-straight-through-estimators.html
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        #return (input > 0).float()
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class STETanh(nn.Module):
    """
    https://www.hassanaskary.com/python/pytorch/deep%20learning/2020/09/19/intuitive-explanation-of-straight-through-estimators.html
    """
    def __init__(self):
        super(STETanh, self).__init__()

    def forward(self, x):
        x = STETanhFunction.apply(x)
        return x

class STEReLU(nn.Module):
    """
    https://www.hassanaskary.com/python/pytorch/deep%20learning/2020/09/19/intuitive-explanation-of-straight-through-estimators.html
    """
    def __init__(self):
        super(STEReLU, self).__init__()

    def forward(self, x):
        x = STEReLUFunction.apply(x)
        return x