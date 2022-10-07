import torch
from qtorch.quant import fixed_point_quantize
from ogb.graphproppred.mol_encoder import AtomEncoder

from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

from models.qlayers.qembedding_fp import QEmbedding_FP

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()

class QAtomEncoder_FP(AtomEncoder):
    def __init__(self, emb_dim,
                 quantizer=fixed_point_quantize, wl=8, fl=4, quant_in=False, quant_out=False):
        super(QAtomEncoder_FP, self).__init__(emb_dim)
        self.wl, self.fl, self.quantizer = wl, fl, quantizer
        self.quant_in, self.quant_out = quant_in, quant_out

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = QEmbedding_FP(dim, emb_dim, quant_in=self.quant_in, quant_out=self.quant_out)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, in_data):
        if self.quant_in: in_data.data = self.quantizer(in_data.float(), self.wl, self.fl, rounding='stochastic').data.long()
        out_data = super().forward(in_data)
        if self.quant_out: out_data.data = self.quantizer(out_data, self.wl, self.fl, rounding='stochastic').data
        return out_data
