import pytorch_lightning as pl
import torch
from ogb.graphproppred.mol_encoder import AtomEncoder
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from pytorch_lightning.metrics.functional import accuracy
from qtorch.quant import fixed_point_quantize
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GINConv

from models import QBatchNorm1d_FP, QAtomEncoder_FP
from models.qlayers import QLinear_FP, QConv2d_FP, QAvgPool2d_FP, QGCNConv_FP, QGINConv_FP
from optimizers import MSGD_FP
from utils import logging
from utils.logging import QLogger
import gc

class QGIN_FP(torch.nn.Module):
    # Source https://zhuanlan.zhihu.com/p/395507027
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gin_conv.html#GINConv
    # https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/conv.py
    def __init__(self, emb_dim, hidden_dim, num_layers,
                 dropout, return_embeds=False,
                 quantizer=fixed_point_quantize, wl=8, fl=4, quant_in=False, quant_out=False):
        super(QGIN_FP, self).__init__()
        self.qargs = dict(quantizer=quantizer, wl=wl, fl=fl,
                          quant_in=quant_in, quant_out=quant_out)

        self.convs = None
        self.bns = None
        self.softmax = None
        self.make_mlp = lambda emb_dim : torch.nn.Sequential(QLinear_FP(emb_dim, 2 * emb_dim, **self.qargs), QBatchNorm1d_FP(2 * emb_dim, **self.qargs),
                                       torch.nn.ReLU(), QLinear_FP(2 * emb_dim, emb_dim, **self.qargs))
        #self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.convs = nn.ModuleList([QGINConv_FP(self.make_mlp(emb_dim), **self.qargs)])
        self.convs.extend([QGINConv_FP(self.make_mlp(emb_dim), **self.qargs) for i in range(num_layers - 2)])
        self.convs.extend([QGINConv_FP(self.make_mlp(emb_dim), **self.qargs)])

        self.bns = torch.nn.ModuleList([QBatchNorm1d_FP(hidden_dim, **self.qargs) for i in range(num_layers - 1)])
        self.softmax = torch.nn.LogSoftmax()
        self.dropout = dropout
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        out = None
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, self.training)
        out = self.convs[len(self.convs) - 1](x, adj_t)  # GCNVonv
        if not self.return_embeds:
            out = self.softmax(out)
        return out

class QGIN_Graph_FP(pl.LightningModule):
    # Source https://zhuanlan.zhihu.com/p/395507027
    def __init__(self, evaluator_name, data_dir='./data', num_classes=10, initializer=None, patience=5, threshold=1.e-4, accumulation_steps=1,
                 l1_decay=1e-4, initializer_arg=1.0, hidden_dim=300, num_layers=2, dropout=0.5,
                 wl_start=8, fl_start=4, quant_in=False, quant_out=False,
                 optimizer=None, **optargs):
        super(QGIN_Graph_FP, self).__init__()

        # Set our init args as class attributes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.evaluator = Evaluator(name=evaluator_name)

        self.l1_decay = l1_decay
        self.data_dir = data_dir
        self.optim = optimizer
        self.optargs = optargs
        self.accumulation_steps = accumulation_steps
        self.patience = patience
        self.threshold = threshold
        self.qargs = dict(quantizer=fixed_point_quantize, wl=wl_start, fl=fl_start,
                          quant_in=quant_in, quant_out=quant_out)

        self.node_encoder = QAtomEncoder_FP(self.hidden_dim, **self.qargs)

        self.gin_node = QGIN_FP(self.hidden_dim, self.hidden_dim,
                            self.num_layers, self.dropout, return_embeds=True, **self.qargs)

        self.pool = None
        self.pool = global_mean_pool

        # Output layer
        self.linear = QLinear_FP(self.hidden_dim, num_classes, **self.qargs)

        self.initializer_arg = initializer_arg
        self.initializer = initializer
        if initializer is not None:
            self.apply(self.initialize_weights)

        # Disable automatic optimization
        self.automatic_optimization = False
        self.stage = "train"
        self.qmap = {}
        self.qlogger = QLogger()

        self.y_true, self.y_pred = [],[]
    def initialize_weights(self, m):
        if hasattr(m, 'weight') and len(m.weight.shape) > 1:
            self.initializer(m.weight, self.initializer_arg)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'):
            # m.bias.data.fill_(0.00)
            stdv = self.initializer_arg
            m.bias.data.uniform_(-stdv, stdv)

    def reset_parameters(self):
        self.gin_node.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, data):

        if self.stage == "train":
            self.qlogger.qlog("qmap", self.qmap)
            self.qlogger.qlog("lr", self.optimizers(use_pl_optimizer=False).param_groups[0]['lr'])

        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, embeding in enumerate(self.node_encoder.atom_embedding_list):
            setattr(embeding, 'wl', self.qmap['qwl'][i])
            setattr(embeding, 'fl', self.qmap['qfl'][i])
        embed = self.node_encoder(x)

        out = None
        x = embed
        skip_bias = 0
        for i in range(len(self.gin_node.convs) - 1):
            j = i + len(self.node_encoder.atom_embedding_list)
            k = i + len(self.node_encoder.atom_embedding_list) + len(self.gin_node.convs)* 6 + skip_bias

            n = 0
            for m, mod in enumerate(self.gin_node.convs[i].modules()):
                if type(mod) == QLinear_FP or type(mod) == QBatchNorm1d_FP:
                    setattr(mod, 'wl', self.qmap['qwl'][j +n + 5 * (j - len(self.node_encoder.atom_embedding_list))])
                    setattr(mod, 'fl', self.qmap['qfl'][j + n + 5 * (j - len(self.node_encoder.atom_embedding_list))])

            x = self.gin_node.convs[i](x, edge_index)
            setattr(self.gin_node.bns[i], 'wl', self.qmap['qwl'][k])
            setattr(self.gin_node.bns[i], 'fl', self.qmap['qfl'][k])
            x = self.gin_node.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.gin_node.dropout, self.gin_node.training)
            skip_bias += 1
        out = self.gin_node.convs[len(self.gin_node.convs) - 1](x, edge_index)  # GCNVonv
        if not self.gin_node.return_embeds:
            out = self.gin_node.softmax(out)
        # return out

        # out = self.gnn_node(embed, edge_index)
        out = self.pool(out, batch)
        setattr(self.linear, 'wl', self.qmap['qwl'][-1])
        setattr(self.linear, 'fl', self.qmap['qfl'][-1])
        out = self.linear(out)
        #exit(0)
        return out

    def print_memstats(self, batch_idx, interval):
        if batch_idx % interval == 0:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print('Using device:', device)
            print('Batch index:', batch_idx)

            #torch.cuda.empty_cache()
            if device.type == 'cuda':
                print(torch.cuda.get_device_name(0))
                print('Memory Usage:')
                print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
                print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
                #print(torch.cuda.memory_stats().keys())
                print('Inactive:  ', round(torch.cuda.memory_stats()['inactive_split_bytes.all.current']/ 1024 ** 3, 1), 'GB')
                #print(torch.cuda.memory_snapshot())

                ctr = 0
                ctr_mem = 0
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                            ctr+=1
                            ctr_mem+=obj.size()
                            #print(type(obj), obj.size())
                    except:
                        pass
                print('Referenced objects:', ctr)
                print('Referenced objects size:', ctr_mem)

    def training_step(self, batch, batch_idx):
        self.print_memstats(batch_idx, 500)
        self.stage = "train"
        opt = self.optimizers(use_pl_optimizer=False)
        out = self(batch)

        is_labeled = True
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y

        # Sparsity inducing L1 regularization.
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(out[is_labeled], batch.y[is_labeled].type_as(out)) + self.l1_decay * l1_norm

        pen, ct = 0, 0
        for p in self.parameters():
            pen += self.qmap['qwl'][ct] / 32 * torch.count_nonzero(p).item() / p.numel()
            ct += 1
        pen = pen / ct
        loss += pen

        loss = loss / self.accumulation_steps
        # loss.backward()
        self.manual_backward(loss, opt)

        if (batch_idx + 1) % self.accumulation_steps == 0:
            opt.step(self.qmap, closure=lambda: loss)
            self.zero_grad(set_to_none=True)

        return loss



    def validation_step(self, batch, batch_idx):
        self.stage = "validation"

        is_labeled = None
        if batch.x.shape[0] == 1:
            pass
        else:
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y

        batch = batch
        out = self(batch)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(out[is_labeled], batch.y[is_labeled].type_as(out))
        y_pred = out.detach().cpu()
        y_true = batch.y.view(out.shape).detach().cpu()
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)


        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss.item(), prog_bar=True)
        #self.log('val_acc', acc, prog_bar=True)

        self.qlogger.qlog('val_loss', loss.item())
        #self.qlogger.qlog('val_acc', acc)
        return loss

    def test_step(self, batch, batch_idx):
        self.stage = "test"
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self, *args, **kwargs):
        input_dict = {"y_true": torch.cat(self.y_true, dim=0).numpy(), "y_pred": torch.cat(self.y_pred, dim=0).numpy()}
        try:
            acc = None
            if "ap" in self.evaluator.eval(input_dict):
                acc = self.evaluator.eval(input_dict)['ap'].item()
            elif "rocauc" in self.evaluator.eval(input_dict):
                acc = self.evaluator.eval(input_dict)['rocauc'].item()
            else:
                raise Exception("Evaluation metric not found")
            self.log('val_acc', acc, prog_bar=True)
            self.qlogger.qlog('val_acc', acc)
        except Exception as e:
            print('y', e, flush=True)
            #acc = 0.0
        self.y_true, self.y_pred = [], []

    def on_test_epoch_end(self, *args, **kwargs):
        self.on_validation_epoch_end(*args, **kwargs)

    def configure_optimizers(self):
        redict = None
        optimizer = self.optim(self.parameters(), **self.optargs)
        if type(self.optim) in [type(torch.optim.SGD), type(MSGD_FP)]:
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=np.arange(0,150,10), gamma=0.1)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=self.patience,
                                                                   verbose=False, threshold=self.threshold)
            redict = {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_loss'
            }
        else:
            redict = {
                'optimizer': optimizer
            }
        return redict

    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            q, dq, qwl, qfl, perm, res, lb, name, sp = [], [], [], [], {}, [], [], [], []
            pnames = []
            for pname, param in self.named_parameters():
                if param.requires_grad:
                    pnames.append(pname)
            ctr = 0
            for pname in pnames:
                q.append(self.qargs['quantizer'])
                qwl.append(self.qargs['wl'])
                qfl.append(self.qargs['fl'])
                res.append(self.optargs['min_resolution'])
                lb.append(self.optargs['min_lookback'])
                name.append(pname.replace('.', ' ').replace('weight', ''))
                sp.append(0)


            self.qmap = {'q': q, 'qwl': qwl, 'qfl': qfl, 'perm': perm, 'res': res, 'lb': lb, 'name': name, 'sp': sp}
            #print(self.qmap, flush=True)
            #exit(0)