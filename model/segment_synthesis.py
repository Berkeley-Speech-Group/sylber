import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from lightning import LightningModule
except:
    lightning = None
from transformers import HubertModel, HubertConfig
from torch.optim.lr_scheduler import LambdaLR
from .flowmatching import Regressor, ConditionalFlowMatcherWrapperRegressor
from utils.segment_utils import get_segment, Thresholder
from utils.lr_schedule import COSLRLAMBDA
from model.quantizer import load_km_quantizer


class RFF(nn.Module):
    def __init__(self, dim, dropout=0.05):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout1= nn.Dropout(dropout)
        self.dropout2= nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        
        x2 = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm(x)
        return x


class MLP(nn.Module):
    def __init__(self,input_dim, output_dim, hidden_dims, dropout=0.05):
        super().__init__()
        #self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        #self.output_layer = nn.Linear(hidden_dims[-1],output_dim)
        
        modules = []
        
        for dim in hidden_dims:
            modules.append(nn.Linear(input_dim,dim))
            modules.append(RFF(dim,dropout))
            input_dim = dim
        
        modules.append(nn.Linear(hidden_dims[-1],output_dim))
        
        self.mlp = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.mlp(x)
        

class SegmentSynthesis(nn.Module):

    def __init__(self,
                 speech_upstream="facebook/hubert-base-ls960",
                 encoding_layer = 9,
                 regressor_configs={},
                 noise_mixer_configs={},
                 segment_online = False,
                 thresholder_configs={},
                 input_configs={},
                 merge_threshold_range=[0.8,0.8],
                 quantizer=None,
                 pitch_amp=5,
                 art_input_configs={},
                 **kwargs,
                ):
        super().__init__()
        self.speech_model = HubertModel(HubertConfig.from_pretrained(speech_upstream, num_hidden_layers=encoding_layer))
        self.speech_model.requires_grad_(False)
        self.enc_dim = self.speech_model.config.hidden_size
        self.input_model = MLP(self.enc_dim, **input_configs)
        self.input_dim = input_configs['output_dim']
        self.encoding_layer = encoding_layer
        
        self.regressor = Regressor(**regressor_configs)
        self.cfm_wrapper = ConditionalFlowMatcherWrapperRegressor(
                regressor = self.regressor,
                sigma=regressor_configs['sigma'],
            )
       
        self.segment_online = segment_online
        if segment_online:
            self.thresholder = Thresholder(**thresholder_configs)
        else:
            self.thresholder = None
        
        self.merge_threshold_range=merge_threshold_range
        if quantizer is not None:
            if residual_quantizer is not None:
                self.quantizer = load_km_quantizer(quantizer,residual_quantizer)
            else:
                self.quantizer = load_km_quantizer(quantizer,normalize=normalize_embed)
            self.quantizer.requires_grad_(False)
        else:
            self.quantizer = None
        self.pitch_amp = pitch_amp
            
    def resynthesize(self,input_values=None,attention_mask=None,features=None,  steps=5,rand_scale=0.0,merge_threshold=0.8,normthreshold=None,
                    prosody_steps=None, prosody_rand_scale=None):
        """
        """      
        if features is None:
            with torch.no_grad():
                self.speech_model.eval()
                target_hidden_states = self.speech_model(input_values, attention_mask=attention_mask).last_hidden_state
            if normthreshold is None:
                normthreshold = self.thresholder.get_threshold().item()
            with torch.no_grad():
                norms = ((target_hidden_states**2).sum(-1)+1e-8)**.5
                
                segments=[get_segment(states.cpu().numpy(), normthreshold, merge_threshold) for states in target_hidden_states]
    
            valid_segments = [segment for segment in segments if len(segment) >0]
            averaged_target_hidden_states = torch.zeros_like(target_hidden_states[:,:,:])
            trg_avg_fts = []
            for b in range(len(target_hidden_states)):
                trg_avg_fts_ = []
                for s,e in segments[b]:
                    with torch.no_grad():
                        trg_avg_ft = target_hidden_states[b][s:e].mean(0)
                        if self.quantizer is not None:
                            with torch.no_grad():
                                q_idxs = self.quantizer.get_indices(trg_avg_ft.unsqueeze(0))
                                q_decoded = self.quantizer.vq.get_output_from_indices(q_idxs.unsqueeze(0)) #/6*trg_avg_ft_norm
                                trg_avg_ft = q_decoded[0]
                        trg_avg_ft = trg_avg_ft
                    averaged_target_hidden_states[b][s:e] = trg_avg_ft
                    trg_avg_fts_.append(trg_avg_ft)
                if len(trg_avg_fts_) != 0:
                    trg_avg_fts.append(torch.stack(trg_avg_fts_))
        else:
            averaged_target_hidden_states = features
            norms = ((features**2).sum(-1))**.5
            normthreshold = 1e-4
            segments = None
        input = self.input_model(averaged_target_hidden_states)
        input[norms<normthreshold]=0.0
        art = self.cfm_wrapper.sample(cond_emb=input,steps=steps,rand_scale=rand_scale)        
        art = art.clone()
        art[...,12] = art[...,12]/self.pitch_amp
        return art,segments
        
    def forward(self,input_values=None, features=None, segments=None, art=None, attention_mask=None, noise=None, **kwargs):
        """
        """        
        if features is None:
            if self.noise_mixer is not None and noise is not None:
                with torch.no_grad():
                    input_values = self.noise_mixer(input_values, noise)
                    
            with torch.no_grad():
                self.speech_model.eval()
                target_hidden_states = self.speech_model(input_values, attention_mask=attention_mask).last_hidden_state
    
            if segments is None:
                assert self.segment_online
                normthreshold = self.thresholder.get_threshold().item()
                with torch.no_grad():
                    norms = ((target_hidden_states**2).sum(-1)+1e-8)**.5
                    if self.merge_threshold_range[0]<self.merge_threshold_range[1]:
                        merge_threshold = np.random.uniform(*self.merge_threshold_range)
                    else:
                        merge_threshold = self.merge_threshold_range[0]
                    segments=[get_segment(states.cpu().numpy(), normthreshold, merge_threshold) for states in target_hidden_states]
    
            
            valid_segments = [segment for segment in segments if len(segment) >0]
            averaged_target_hidden_states = torch.zeros_like(target_hidden_states[:,:,:])
            trg_avg_fts = []
            for b in range(len(target_hidden_states)):
                trg_avg_fts_ = []
                for s,e in segments[b]:
                    trg_avg_ft = target_hidden_states[b][s:e].mean(0)
                    trg_avg_ft = trg_avg_ft
                    averaged_target_hidden_states[b][s:e] = trg_avg_ft
                    trg_avg_fts_.append(trg_avg_ft)
                if len(trg_avg_fts_) != 0:
                    trg_avg_fts.append(torch.stack(trg_avg_fts_))
            input = self.input_model(averaged_target_hidden_states)
            input[norms<normthreshold]=0.0
            features = averaged_target_hidden_states
        else:
            norms = ((features**2).sum(-1))**.5
            if self.quantizer is not None:
                with torch.no_grad():
                    q_idxs = self.quantizer.get_indices(features)
                    features = self.quantizer.decode(q_idxs)
                    features[norms<1e-4] = 0.0
            input = self.input_model(features)
            normthreshold=1e-4
            input[norms<normthreshold]=0.0
        
        art[...,12] *= self.pitch_amp
        outputs = {'cfm_loss': cfm_loss}
        
        return outputs

class SegmentSynthesisTrainer(LightningModule):

    def __init__(self, loss_coefs,lr=0.0001, warmup_steps=5000, total_steps=500000, min_factor=0.05, 
                 **model_configs):
        super().__init__()
        self.loss_coefs = loss_coefs
        
        self.net = SegmentSynthesis(**model_configs).to(torch.float)
        self.lr = lr
        self.total_steps =total_steps
        self.warmup_steps = warmup_steps
        self.min_factor = min_factor
        
    def forward(self, **kwargs):
        return self.net(**kwargs)
    
    def training_step(self, batch, batch_idx):
        outputs = self.net(**batch)
        
        loss_val = 0
        for coef_name, coef_val in self.loss_coefs.items():
            if coef_name in outputs.keys():
                loss_val += coef_val * outputs[coef_name]
                self.log(f'train_{coef_name}', outputs[coef_name],sync_dist=True)
        self.log(f'train_loss', loss_val)
        
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
        lr_lambda = COSLRLAMBDA(self.warmup_steps, self.total_steps, self.min_factor)
        sch = LambdaLR(opt, lr_lambda)
        return [opt],[{"scheduler": sch, "interval": "step"}]
