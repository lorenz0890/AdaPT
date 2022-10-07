from qtorch.quant import fixed_point_quantize
from torch_geometric.nn import GCNConv

class QGCNConv_FP(GCNConv):
    def __init__(self, in_channels, out_channels, improved=False, bias=True,
                 quantizer=fixed_point_quantize, wl=8, fl=4, quant_in=False, quant_out=False):
        super(QGCNConv_FP, self).__init__(in_channels, out_channels, improved=improved, bias=bias)
        self.wl, self.fl, self.quantizer = wl, fl, quantizer
        self.quant_in, self.quant_out = quant_in, quant_out

    def forward(self, in_data, edge_index,
                edge_weight = None):
        if self.quant_in: in_data.data = self.quantizer(in_data, self.wl, self.fl, rounding='stochastic').data
        out_data = super().forward(in_data, edge_index,
                edge_weight)
        if self.quant_out: out_data.data = self.quantizer(out_data, self.wl, self.fl, rounding='stochastic').data
        return out_data
