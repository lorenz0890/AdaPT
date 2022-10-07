import math

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import accuracy
from qtorch.quant import fixed_point_quantize
from torch import nn
from torch.nn import functional as F

from optimizers import MSGD_FP
from utils import logging
from utils.logging import QLogger
from .qlayers import QLinear_FP, QConv2d_FP, QAvgPool2d_FP, QBatchNorm2d_FP


# Ported from
# Source:
# https://github.com/ICIdsl/muppet/blob/master/models/cifar/resnet.py
def QConv3x3_FP(in_planes, out_planes, stride=1, **qargs):
    """
    3x3 convolution with padding
    :param in_planes:
    :param out_planes:
    :param stride:
    :param qargs:
    :return:
    """
    mapped = False
    return QConv2d_FP(in_planes, out_planes, kernel_size=3, stride=stride,
                      padding=1, bias=False, **qargs)


class QBasicBlock_FP(nn.Module):
    mapped = False
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, **qargs):
        super(QBasicBlock_FP, self).__init__()
        self.conv1 = QConv3x3_FP(inplanes, planes, stride, **qargs)
        self.bn1 = QBatchNorm2d_FP(planes, **qargs)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = QConv3x3_FP(planes, planes, **qargs)
        self.bn2 = QBatchNorm2d_FP(planes, **qargs)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class QBottleneck_FP(nn.Module):
    expansion = 4
    mapped = False

    def __init__(self, inplanes, planes, stride=1, downsample=None, **qargs):
        super(QBottleneck_FP, self).__init__()
        self.conv1 = QConv2d_FP(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False, **qargs)
        self.bn1 = QBatchNorm2d_FP(planes, **qargs)
        self.conv2 = QConv2d_FP(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, **qargs)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = QConv2d_FP(planes, planes * 4, kernel_size=1, stride=1, padding=0, bias=False, **qargs)
        self.bn3 = QBatchNorm2d_FP(planes * 4, **qargs)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class QResNet_MNIST_FP(pl.LightningModule):

    def __init__(self, data_dir='./data', depth=20, num_classes=1000, patience=5, threshold=1.e-4, accumulation_steps=1,
                 l1_decay=1e-4, initializer=None, initializer_arg=1.0,
                 wl_start=8, fl_start=4, quant_in=False, quant_out=False, optimizer=None, **optargs):
        super(QResNet_MNIST_FP, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = QBottleneck_FP if depth >= 44 else QBasicBlock_FP

        self.ct = 0

        # Set our init args as class attributes
        self.l1_decay = l1_decay
        self.data_dir = data_dir
        self.optim = optimizer
        self.accumulation_steps = accumulation_steps
        self.patience = patience
        self.threshold = threshold
        self.optargs = optargs
        self.qargs = dict(quantizer=fixed_point_quantize, wl=wl_start, fl=fl_start,
                          quant_in=quant_in, quant_out=quant_out)

        self.inplanes = 16
        self.conv1 = QConv2d_FP(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False,
                                **self.qargs)
        self.bn1 = QBatchNorm2d_FP(16, **self.qargs)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = QAvgPool2d_FP(8, stride=None, padding=0, ceil_mode=False, count_include_pad=True, **self.qargs)
        self.fc = QLinear_FP(64 * block.expansion, num_classes, **self.qargs)

        for m in self.modules():
            if isinstance(m, QConv2d_FP):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
            m.bias.data.uniform_(-stdv, stdv)  # (0.00)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                QConv2d_FP(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, padding=0, bias=False,
                           **self.qargs),
                QBatchNorm2d_FP(planes * block.expansion, **self.qargs),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, **self.qargs))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, **self.qargs))

        return nn.Sequential(*layers)

    def __forward_helper(self, p, ctr):
        if QBasicBlock_FP == type(p) or QBottleneck_FP == type(p) or QConv3x3_FP == type(p):
            p.mapped = True
            for i, p_sub in enumerate(p.modules()):
                if QBasicBlock_FP == type(p_sub) or QBottleneck_FP == type(p_sub) or QConv3x3_FP == type(p_sub):
                    # print(p_sub, p_sub.mapped, flush=True)
                    if p_sub.mapped: continue
                self.__forward_helper(p_sub, ctr)
        elif (QAvgPool2d_FP != type(p) and nn.ReLU != type(p) and
              nn.Sequential != type(p) and QResNet_MNIST_FP != type(p)):
            p.wl, p.fl = self.qmap['qwl'][ctr[0]], self.qmap['qfl'][ctr[0]]
            ctr[0] += 1
        elif (QAvgPool2d_FP == type(p)):
            p.wl, p.fl = self.qmap['qwl'][ctr[0]], self.qmap['qfl'][ctr[0]]
        # ctr[0] += 1
        # print(p, p.wl, flush=True)

    def forward(self, x):
        # Tracing
        if self.stage == "train":
            self.qlogger.qlog("qmap", self.qmap)
            self.qlogger.qlog("lr", self.optimizers(use_pl_optimizer=False).param_groups[0]['lr'])

        self.conv1.wl, self.conv1.fl = self.qmap['qwl'][0], self.qmap['qfl'][0]
        x = self.conv1(x)

        self.bn1.wl, self.bn1.fl = self.qmap['qwl'][2], self.qmap['qfl'][2]
        x = self.bn1(x)
        x = self.relu(x)  # 32x32

        ctr = [0]
        for p in self.layer1: self.__forward_helper(p, ctr)
        for p in self.layer2: self.__forward_helper(p, ctr)
        for p in self.layer3: self.__forward_helper(p, ctr)
        for p in self.layer1: self.__set_mapping(p, False)
        for p in self.layer2: self.__set_mapping(p, False)
        for p in self.layer3: self.__set_mapping(p, False)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        i = len(self.qmap['qwl']) - 3
        self.avgpool.wl, self.avgpool.fl = self.qmap['qwl'][i], self.qmap['qfl'][i]
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        i = len(self.qmap['qwl']) - 1
        self.fc.wl, self.fc.fl = self.qmap['qwl'][i], self.qmap['qfl'][i]
        x = self.fc(x)

        return x

    def training_step(self, batch, batch_idx):
        self.stage = "train"
        opt = self.optimizers(use_pl_optimizer=False)
        x, y = batch
        logits = self(x)

        # Sparsity inducing L1 regularization.
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        loss = F.cross_entropy(logits, y) + self.l1_decay * l1_norm

        '''
        pen, ct = 0, 0
        for p in self.parameters():
            pen += self.qmap['qwl'][ct] / 32 * torch.count_nonzero(p).item() / p.numel()
            ct += 1
        pen = pen / ct
        loss += pen
        '''

        # loss = F.cross_entropy(logits, y)
        loss = loss / self.accumulation_steps
        self.manual_backward(loss, opt)

        if (batch_idx + 1) % self.accumulation_steps == 0:
            opt.step(qmap=self.qmap, closure=lambda: loss)
            self.zero_grad(set_to_none=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.stage = "validation"

        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        # Calling self.log will surface up scalars for you in TensorBoard

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        self.qlogger.qlog('val_loss', loss.item())
        self.qlogger.qlog('val_acc', acc.item())

        return loss

    def test_step(self, batch, batch_idx):
        self.stage = "test"
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        redict = None
        optimizer = self.optim(self.parameters(), **self.optargs)
        if type(self.optim) in [type(torch.optim.SGD), type(MSGD_FP)]:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=np.arange(10,12), gamma=0.1)
            #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=self.patience,
            #                                                       verbose=False, threshold=self.threshold)
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

    def __set_mapping(self, p, value):
        if QBasicBlock_FP == type(p) or QBottleneck_FP == type(p) or QConv3x3_FP == type(p):
            p.mapped = value
        for i, p_sub in enumerate(p.modules()):
            if QBasicBlock_FP == type(p_sub) or QBottleneck_FP == type(p_sub) or QConv3x3_FP == type(p_sub):
                if p_sub.mapped == value: continue
                self.__set_mapping(p_sub, value)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            q, dq, qwl, qfl, perm, res, lb, name, sp = [], [], [], [], {}, [], [], [], []
            ctr = 0
            for pname, param in self.named_parameters():
                if param.requires_grad:
                    q.append(self.qargs['quantizer'])
                    qwl.append(8)
                    qfl.append(4)
                    res.append(self.optargs['min_resolution'])
                    lb.append(self.optargs['min_lookback'])
                    name.append(pname.replace('.', ' ').replace('weight', ''))
                    sp.append(0)

            self.qmap = {'q': q, 'qwl': qwl, 'qfl': qfl, 'perm': perm, 'res': res, 'lb': lb, 'name': name, 'sp' : sp}
