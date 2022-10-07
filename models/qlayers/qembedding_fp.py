import torch
from qtorch.quant import fixed_point_quantize


class QEmbedding_FP(torch.nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
                 norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None,
                 device=None, dtype=None,
                 quantizer=fixed_point_quantize, wl=8, fl=4, quant_in=False, quant_out=False):
        super(QEmbedding_FP, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm,
                                            norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse,
                                            _weight=_weight)
        self.wl, self.fl, self.quantizer = wl, fl, quantizer
        self.quant_in, self.quant_out = quant_in, quant_out

    def forward(self, in_data):
        if self.quant_in: in_data.data = self.quantizer(in_data.float(), self.wl, self.fl, rounding='stochastic').data.long()
        out_data = super().forward(in_data)
        if self.quant_out: out_data.data = self.quantizer(out_data, self.wl, self.fl, rounding='stochastic').data
        return out_data
