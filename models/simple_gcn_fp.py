import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import accuracy
from qtorch.quant import fixed_point_quantize
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

from models.qlayers import QLinear_FP, QConv2d_FP, QAvgPool2d_FP, QGCNConv_FP
from optimizers import MSGD_FP
from utils import logging
from utils.logging import QLogger


class QSimple_GCN_FP(pl.LightningModule):
    def __init__(self, data_dir='./data', num_classes=10, initializer=None, patience=5, threshold=1.e-4, accumulation_steps=1,
                 l1_decay=1e-4, initializer_arg=1.0,
                 wl_start=32, fl_start=16, quant_in=False, quant_out=False, optimizer=None, **optargs):
        super(QSimple_GCN_FP, self).__init__()

        # Set our init args as class attributes
        self.l1_decay = l1_decay
        self.data_dir = data_dir
        self.optim = optimizer
        self.optargs = optargs
        self.accumulation_steps = accumulation_steps
        self.patience = patience
        self.threshold = threshold
        self.qargs = dict(quantizer=fixed_point_quantize, wl=wl_start, fl=fl_start,
                          quant_in=quant_in, quant_out=quant_out)
        self.layers = nn.Sequential(
            # Extractor
            QGCNConv_FP(1433, 16, **self.qargs),
            nn.ReLU(),
            nn.Dropout(),
            # Classifier
            QGCNConv_FP(16, 7, **self.qargs),
        )
        self.initializer_arg = initializer_arg
        self.initializer = initializer
        if initializer is not None:
            self.apply(self.initialize_weights)

        # Disable automatic optimization
        self.automatic_optimization = False
        self.stage = "train"
        self.qmap = {}
        self.qlogger = QLogger()

    def initialize_weights(self, m):
        if hasattr(m, 'weight') and len(m.weight.shape) > 1:
            self.initializer(m.weight, self.initializer_arg)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'):
            # m.bias.data.fill_(0.00)
            stdv = self.initializer_arg
            m.bias.data.uniform_(-stdv, stdv)

    def forward(self, data):

        # Tracing
        if self.stage == "train":
            self.qlogger.qlog("qmap", self.qmap)
            self.qlogger.qlog("lr", self.optimizers(use_pl_optimizer=False).param_groups[0]['lr'])

        x, edge_index = data.x, data.edge_index
        for layer in self.layers:
            if type(layer) == type(self.layers[0]):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        #print(x)
        #x = self.conv1(x, edge_index)
        #x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        #x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

        """
        # Extractor
        for i in range(0, 8):
            if self.qmap['perm'][i] is not None:
                j = self.qmap['perm'][i]
                setattr(self.layers[i], 'wl', self.qmap['qwl'][j])
                setattr(self.layers[i], 'fl', self.qmap['qfl'][j])
            x = self.layers[i](x)
        x = torch.flatten(x, 1)

        # Classifier
        for i in range(8, 11):
            if self.qmap['perm'][i] is not None:
                j = self.qmap['perm'][i]
                setattr(self.layers[i], 'wl', self.qmap['qwl'][j])
                setattr(self.layers[i], 'fl', self.qmap['qfl'][j])
            x = self.layers[i](x)
        logits = x
        probs = F.softmax(logits, dim=1)

        return probs
        """

    def training_step(self, batch, batch_idx):
        self.stage = "train"
        opt = self.optimizers(use_pl_optimizer=False)
        out = self(batch)

        # Sparsity inducing L1 regularization.
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask]) + self.l1_decay * l1_norm

        pen, ct = 0, 0
        for p in self.parameters():
            pen += self.qmap['qwl'][ct] / 32 * torch.count_nonzero(p).item() / p.numel()
            ct += 1
        pen = pen / ct
        loss += pen

        loss = loss / self.accumulation_steps
        #loss.backward()
        self.manual_backward(loss, opt)

        if (batch_idx + 1) % self.accumulation_steps == 0:
            opt.step(qmap=self.qmap, closure=lambda: loss)
            self.zero_grad(set_to_none=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.stage = "validation"
        #print(batch, flush=True)
        x = batch
        logits = self(x)
        #print(x.__dict__['x'], flush=True)
        #loss = F.log_softmax(x.__dict__['x'], dim=1)
        loss = F.nll_loss(logits[batch.train_mask], batch.y[batch.train_mask])
        preds = torch.argmax(logits, dim=1)
        correct = int(preds[batch.test_mask].eq(batch.y[batch.test_mask]).sum().item())
        acc = correct / int(batch.test_mask.sum())

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss.item(), prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        self.qlogger.qlog('val_loss', loss.item())
        self.qlogger.qlog('val_acc', acc)
        return loss

    def test_step(self, batch, batch_idx):
        self.stage = "test"
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

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
            j = 0
            for i, p in enumerate(self.modules()):
                if 'bias' in p.__dict__['_parameters']:  # getattr this
                    if 'quantizer' in p.__dict__:
                        q.append(p.__dict__['quantizer'])
                        qwl.append(p.__dict__['wl'])
                        qfl.append(p.__dict__['fl'])
                        res.append(self.optargs['min_resolution'])
                        lb.append(self.optargs['min_lookback'])
                        name.append(str(type(p)).split('.')[-1][:-2])
                        sp.append(0)
                    else:
                        q.append(None)
                        qwl.append(None)
                        qfl.append(None)
                        res.append(None)
                        lb.append(None)
                        name.append(None)
                        sp.append(0)

                if 'weight' in p.__dict__['_parameters']:
                    if 'quantizer' in p.__dict__:
                        q.append(p.__dict__['quantizer'])
                        qwl.append(p.__dict__['wl'])
                        qfl.append(p.__dict__['fl'])
                        res.append(self.optargs['min_resolution'])
                        lb.append(self.optargs['min_lookback'])
                        name.append(str(type(p)).split('.')[-1][:-2])
                        sp.append(0)
                        #sp.append(torch.count_nonzero(p.__dict__['_parameters']['weight'].data).item())
                    else:
                        q.append(None)
                        qwl.append(None)
                        qfl.append(None)
                        res.append(None)
                        lb.append(None)
                        name.append(None)
                        sp.append(0)

                if 'weight' in p.__dict__['_parameters'] or 'bias' in p.__dict__['_parameters']:
                    perm[i] = j
                else:
                    perm[i] = None

            self.qmap = {'q': q, 'qwl': qwl, 'qfl': qfl, 'perm': perm, 'res': res, 'lb': lb, 'name': name, 'sp': sp}
