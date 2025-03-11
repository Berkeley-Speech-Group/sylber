import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import init
from ..utils.lr_schedule import COSLRLAMBDA
from .sylber import Sylber


class SylberTrainer(LightningModule):

    def __init__(self, loss_coefs,lr=0.0001, warmup_steps=5000, total_steps=500000,
                 min_factor=1.0,hold_steps=0, accumulate_grad_batches=1,
                 **model_configs):
        super().__init__()
        self.loss_coefs = loss_coefs
        
        self.net = Sylber(**model_configs).to(torch.float)
        self.lr = lr
        self.total_steps =total_steps
        self.warmup_steps = warmup_steps
        self.min_factor = min_factor
        self.hold_steps = hold_steps
        self.accumulate_grad_batches = accumulate_grad_batches
        
    def forward(self, **kwargs):
        return self.net(**kwargs)
        
    def training_step(self, batch, batch_idx):
        
        if (self.global_step%self.accumulate_grad_batches) ==0:
            self.net.ema_step()
        outputs = self.net(**batch)
        
        loss_val = 0
        for coef_name, coef_val in self.loss_coefs.items():
            if coef_name in outputs.keys():
                loss_val += coef_val * outputs[coef_name]
                self.log(f'train_{coef_name}', outputs[coef_name],sync_dist=True)
        self.log(f'train_loss', loss_val)
        if self.net.thresholder is not None:
            self.log('normthreshold', self.net.thresholder.get_threshold().item())
        return loss_val

        
    def validation_step(self, batch, batch_idx):
        outputs = self.net(**batch)
        loss_val = 0
        
        for coef_name, coef_val in self.loss_coefs.items():
            if coef_name in outputs.keys():
                loss_val += coef_val * outputs[coef_name]
                self.log(f'val_{coef_name}', outputs[coef_name],sync_dist=True)
        self.log(f'val_loss', loss_val,sync_dist=True)
            
        return loss_val

    def configure_optimizers(self):
        
        opt_fun = torch.optim.AdamW
        opt = opt_fun(self.net.parameters(),lr=self.lr,eps=1e-4, betas=(0.9, 0.95), weight_decay = 0.1)
        lr_lambda = COSLRLAMBDA(self.warmup_steps, self.total_steps, self.min_factor,  self.hold_steps)
        sch = LambdaLR(opt, lr_lambda)
        return [opt],[{"scheduler": sch, "interval": "step"}]
