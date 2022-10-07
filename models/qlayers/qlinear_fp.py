from qtorch.quant import fixed_point_quantize
from torch import nn


class QLinear_FP(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                 quantizer=fixed_point_quantize, wl=8, fl=4, quant_in=False, quant_out=False):
        super(QLinear_FP, self).__init__(in_features, out_features, bias=bias)
        self.wl, self.fl, self.quantizer = wl, fl, quantizer
        self.quant_in, self.quant_out = quant_in, quant_out

    def forward(self, in_data):
        if self.quant_in: in_data.data = self.quantizer(in_data, self.wl, self.fl, rounding='stochastic').data
        out_data = super().forward(in_data)
        if self.quant_out: out_data.data = self.quantizer(out_data, self.wl, self.fl, rounding='stochastic').data
        return out_data
