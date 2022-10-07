import gc
import math

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.nn import functional as F

from optimizers import MSGD_FP
from utils.logging import QLogger


class AlexNet_ImageNet(pl.LightningModule):
    def __init__(self, data_dir='./data', num_classes=10, patience=5, accumulation_steps=1, threshold=1.e-4,
                 l1_decay=1e-4, initializer=None, initializer_arg=0.5,
                 optimizer=None, **optargs):
        super(AlexNet_ImageNet, self).__init__()

        # Set our init args as class attributes
        self.l1_decay = l1_decay
        self.data_dir = data_dir
        self.optim = optimizer
        self.accumulation_steps = accumulation_steps
        self.patience = patience
        self.threshold = threshold
        self.optargs = optargs
        '''
        self.layers = nn.Sequential(
            # Extractor
            
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2), #padding 2, stride 4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )
        '''
        self.layers = nn.Sequential(
            # Extractor
            nn.Conv2d(3, 64, kernel_size=8, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,num_classes)
        )
        self.initializer_arg = initializer_arg
        self.initializer = initializer
        if initializer is not None:
            self.apply(self.initialize_weights)

        # Disable automatic optimization
        self.automatic_optimization = False
        self.stage = "train"
        self.qlogger = QLogger()

    def initialize_weights(self, m):
        if hasattr(m, 'weight') and len(m.weight.shape) > 1:
            self.initializer(m.weight, self.initializer_arg)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        '''
        if self.stage == "train":
            self.qlogger.qlog("lr", self.optimizers(use_pl_optimizer=False).param_groups[0]['lr'])
        # Extractor
        for i in range(0, len(self.layers) - 1):
            #print(i, x.shape, flush=True)
            if i == 13:
                #print(x.shape)
                x = x.view(x.size(0), 256 * 6 * 6)
            x = self.layers[i](x)
        x = x.view(x.size(0), -1)
        #exit()
        # Classifier
        i = len(self.layers) - 1
        x = self.layers[i](x)
        return x
        '''
        #self.lrs.append(self.optimizers(use_pl_optimizer=False).param_groups[0]['lr'])
        # Extractor
        #
        #print(x.shape, flush=True)
        for i in range(0, len(self.layers) - 1):
            #print(i, x.shape, flush=True)
            if i == 14:
                x = x.view(x.size(0), 256 * 6 * 6)
                #x = x.view(x.size(0), -1)
            x = self.layers[i](x)
        #print(x.shape, flush=True)
        #x = x.view(x.size(0), -1)
        #print(x.shape, flush=True)
        # Classifier
        i = len(self.layers) - 1
        x = self.layers[i](x)
        return x

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
        torch.cuda.empty_cache()
        gc.collect()

        #self.print_memstats(batch_idx, 500)

        self.stage = "train"
        opt = self.optimizers(use_pl_optimizer=False)
        x, y = batch
        logits = self(x)

        # Sparsity inducing L1 regularization.
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        loss = F.cross_entropy(logits, y) + self.l1_decay * l1_norm

        # loss = F.cross_entropy(logits, y)
        loss = loss / self.accumulation_steps
        self.manual_backward(loss, opt)

        if (batch_idx + 1) % self.accumulation_steps == 0:
            opt.step(closure=lambda: loss)
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

        self.qlogger.qlog("val_loss", loss.item())
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
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=self.patience,
                                                                   threshold=self.threshold,
                                                                   verbose=False)
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)
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
        pass
