from qtorch.quant import fixed_point_quantize
from torch import nn


class QAvgPool2d_FP(nn.AvgPool2d):
    def __init__(self, kernel_size, stride, padding, ceil_mode, count_include_pad,
                 quantizer=fixed_point_quantize, wl=8, fl=4, quant_in=False, quant_out=False):
        super(QAvgPool2d_FP, self).__init__(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                                            count_include_pad=count_include_pad)
        self.wl, self.fl, self.quantizer = wl, fl, quantizer
        self.quant_in, self.quant_out = quant_in, quant_out

    def forward(self, in_data):
        if self.quant_in: in_data.data = self.quantizer(in_data, self.wl, self.fl, rounding='stochastic').data
        out_data = super().forward(in_data)
        if self.quant_out: out_data.data = self.quantizer(out_data, self.wl, self.fl, rounding='stochastic').data
        return out_data
