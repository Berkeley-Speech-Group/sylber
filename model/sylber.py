import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from lightning import LightningModule
except:
    lightning = None
from transformers import HubertModel, HubertConfig, BertModel, BertConfig
from .ema_module import EMAModule
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import init
from utils.segment_utils import get_segment, Thresholder
from utils.lr_schedule import COSLRLAMBDA
from utils.noise_utils import NoiseMixer
import torchaudio

    
def apply_xavier_init(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d)):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

class Segmenter():

    def __init__(self,
                 model_ckpt=None,
                 speech_upstream="facebook/hubert-base-ls960",
                 ema_decay=0.999,
                 encoding_layer = 9,
                 merge_threshold=0.8,
                 norm_threshold=2.6,
                 device='cuda',
                 **kwargs,
                ):
        super().__init__()
        self.speech_model = HubertModel(HubertConfig.from_pretrained(speech_upstream, num_hidden_layers=encoding_layer))
        
        self.enc_dim = self.speech_model.config.hidden_size
        self.encoding_layer = encoding_layer

        if model_ckpt is not None:
            state_dict = torch.load(model_ckpt, map_location='cpu')
            self.speech_model.load_state_dict(state_dict, strict=False )
            print("Pre-trained checkpoint loaded")
        self.speech_model = self.speech_model.eval().to(device)
        self.device = device
        self.norm_threshold = norm_threshold
        self.merge_threshold=merge_threshold

    def __call__(self, wav_file):
        """
        Process single wav file or a list of wav files through the model
        
        Args:
            wav_file: Path to a single wav file or a list of wav file paths
            
        Returns:
            For single file: Dictionary with segments and segment_features
            For multiple files: List of dictionaries, each with segments and segment_features
        """
        is_batch = isinstance(wav_file, list)
        wav_files = wav_file if is_batch else [wav_file]
        
        # Load and preprocess all wav files
        batch_wavs = []
        orig_lengths = []
        max_length = 0
        
        for file in wav_files:
            wav, sr = torchaudio.load(file)
            if sr != 16000:
                wav = torchaudio.transforms.Resample(sr, 16000)(wav)
            wav = (wav - wav.mean()) / wav.std()
            orig_lengths.append(wav.shape[1])
            batch_wavs.append(wav)
            max_length = max(max_length, wav.shape[1])
        
        # Pad wavs to the same size
        padded_wavs = []
        attention_masks = []
        
        for wav, _ in zip(batch_wavs, orig_lengths):
            padding = max_length - wav.shape[1]
            if padding > 0:
                padded_wav = torch.nn.functional.pad(wav, (0, padding))
                # Create attention mask (1 for real data, 0 for padding)
                attention_mask = torch.ones(wav.shape[1], dtype=torch.long)
                attention_mask = torch.nn.functional.pad(attention_mask, (0, padding), value=0)
            else:
                padded_wav = wav
                attention_mask = torch.ones(wav.shape[1], dtype=torch.long)
            
            padded_wavs.append(padded_wav)
            attention_masks.append(attention_mask)
        
        batch_tensor = torch.cat(padded_wavs, dim=0).to(self.device)
        attention_mask = torch.stack(attention_masks).to(self.device)
                
        with torch.no_grad():
            self.speech_model.eval()
            hidden_states = self.speech_model(batch_tensor, attention_mask=attention_mask).last_hidden_state
                    
        # Process results serially
        hidden_states = hidden_states.cpu().numpy()
        all_segments = [get_segment(states, self.norm_threshold, self.merge_threshold) for states in hidden_states]
        
        outputs = []
        for i, segments in enumerate(all_segments):
            states = hidden_states[i]
            result = {
                'segments': segments * 1.0 / 50,
                'segment_features': np.stack([states[s:e].mean(0) for s, e in segments]) if len(segments) > 0 else np.array([]),
                'hidden_states': states
            }
            outputs.append(result)
        
        return outputs if is_batch else outputs[0]
        
class Sylber(nn.Module):

    def __init__(self,
                 speech_upstream="facebook/hubert-base-ls960",
                 load_hubert=False,
                 ema_decay=0.999,
                 encoding_layer = 9,
                 do_noise_augment=False,
                 extract_target_from=None,
                 noise_mixer_configs={},
                 mask_prob=0.0,
                 segment_online = False,
                 thresholder_configs={},
                 min_mask_n = 0,
                 ema_ckpt=None,
                 merge_threshold_range=[0.5,0.7],
                 max_mask_set=1,
                 use_train_thrupdate=False,
                 **kwargs,
                ):
        super().__init__()
        if load_hubert:
            self.speech_model = HubertModel.from_pretrained(speech_upstream, num_hidden_layers=encoding_layer)
        else:
            self.speech_model = HubertModel(HubertConfig.from_pretrained(speech_upstream, num_hidden_layers=encoding_layer))
            
        self.encoding_layer = encoding_layer
        self.enc_dim = self.speech_model.config.hidden_size
        
        self.ema = None
        self.ema_decay = ema_decay
        self.ema_ckpt = ema_ckpt
        
        
        if do_noise_augment:
            self.noise_mixer = NoiseMixer(**noise_mixer_configs)
        else:
            self.noise_mixer = None
        
        
        self.segment_online = segment_online
        if segment_online:
            self.thresholder = Thresholder(**thresholder_configs)
        else:
            self.thresholder = None

        self.mask_prob = mask_prob
        self.min_mask_n = min_mask_n
        self.max_mask_set=max_mask_set

        self.merge_threshold_range=merge_threshold_range
        self.use_train_thrupdate=use_train_thrupdate
        

    def ema_step(self):
        if self.ema is None:
            self.ema = EMAModule(
                self.speech_model,
                ema_decay=self.ema_decay,
            )
            
            if self.ema_ckpt is not None:
        
                ema_dict = torch.load(self.ema_ckpt, map_location='cpu')
                self.ema.model.load_state_dict(ema_dict['ema'])
        else:
            self.ema.step(self.speech_model)
            
    def segment(self, input_values=None, features=None, attention_mask=None, mergethreshold=None, normthreshold=None, **kwargs):
        """
        """        
        if features is None:
            with torch.no_grad():
                self.speech_model.eval()
                outputs = self.speech_model(input_values, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state
            features = hidden_states
        hidden_states = features
        if normthreshold is None:
            assert self.segment_online
            normthreshold = self.thresholder.get_threshold().item()
        with torch.no_grad():
            if mergethreshold is None:
                if self.merge_threshold_range[0]<self.merge_threshold_range[1]:
                    merge_threshold = np.random.uniform(*self.merge_threshold_range)
                else:
                    merge_threshold = self.merge_threshold_range[0]
            else:
                merge_threshold = mergethreshold
            segments=[get_segment(states.cpu().numpy(), normthreshold, merge_threshold) for states in hidden_states]

        avg_fts = []
        avg_ft_attention_masks = []
        for b in range(len(hidden_states)):
            avg_fts_ = []
            for s,e in segments[b]:
                avg_ft = features[b][s:e].mean(0)
                avg_fts_.append(avg_ft)
            if len(avg_fts_) != 0:
                avg_fts.append(torch.stack(avg_fts_))
            else:
                avg_fts.append(torch.zeros_like(features[0,:1,:]))
            avg_ft_attention_masks.append(torch.ones(len(avg_fts_), device=features.device))
            
        avg_fts = nn.utils.rnn.pad_sequence(avg_fts, batch_first=True, padding_value=0.0)
        avg_ft_attention_masks = nn.utils.rnn.pad_sequence(avg_ft_attention_masks, batch_first=True, padding_value=0)   
        
        return features, segments, avg_fts
        
        
    def forward(self,input_values, segments=None, attention_mask=None, noise=None, output_states=False,**kwargs):
        """
        """        
        with torch.no_grad():
            if self.ema is None:
                self.ema_step()
            self.ema.model.eval()
            target_hidden_states = self.ema.model(input_values, attention_mask=attention_mask).last_hidden_state

        
        if segments is None:
            assert self.segment_online
            normthreshold = self.thresholder.get_threshold().item()
            with torch.no_grad():
                norms = ((target_hidden_states**2).sum(-1)+1e-8)**.5
                if not self.use_train_thrupdate:
                    self.thresholder.update_stats(norms[norms>=normthreshold],
                                                  norms[norms<normthreshold])
                else:
                    self.thresholder.update_stats(norms[norms>=normthreshold],
                                                  None)
                norm_mask = norms>=normthreshold
                if self.merge_threshold_range[0]<self.merge_threshold_range[1]:
                    merge_threshold = np.random.uniform(*self.merge_threshold_range)
                else:
                    merge_threshold = self.merge_threshold_range[0]
                segments=[get_segment(states.cpu().numpy(), normthreshold, merge_threshold) for states in target_hidden_states]
        masks = [] 
        mask_time_indices = torch.zeros_like(target_hidden_states[...,0])
        total_mask_n = 0
        for b in range(len(target_hidden_states)):
            if len(segments[b])==0:
                continue
            mask = torch.zeros(len(segments[b]), device=target_hidden_states.device)
            mask_n = (np.random.uniform(size=len(segments[b]))<self.mask_prob).sum()
            mask_n = max(self.min_mask_n, mask_n)
            total_mask_n += mask_n
            if mask_n >0:
                mask_idxs = np.random.randint(len(segments[b]),size=mask_n)
                for mask_idx in mask_idxs:
                    mask_set= np.random.randint(1,self.max_mask_set+1)
                    mask[mask_idx:min(len(mask),mask_idx+mask_set)] = 1
                    mask_time = [segments[b][mask_idx][0],
                                 segments[b][min(len(mask),mask_idx+mask_set)-1][1]]
                    mask_time_indices[b][mask_time[0]:mask_time[1]]=1
            masks.append(mask)
        mask_time_indices = mask_time_indices>0

        if self.noise_mixer is not None:
            assert noise is not None
            with torch.no_grad():
                input_values = self.noise_mixer(input_values, noise)


        hidden_states = self.speech_model(input_values, 
                                          attention_mask=attention_mask,
                                          mask_time_indices=mask_time_indices).last_hidden_state

        if self.use_train_thrupdate:
            with torch.no_grad():
                train_norms = ((hidden_states**2).sum(-1)+1e-8)**.5
                self.thresholder.update_stats(None, train_norms[~norm_mask])
                
        averaged_target_hidden_states = torch.zeros_like(target_hidden_states)
        avg_fts = []
        trg_avg_fts = []
        avg_ft_attention_masks = []
        for b in range(len(target_hidden_states)):
            trg_avg_fts_ = []
            avg_fts_ = []
            for s,e in segments[b]:
                if np.random.uniform()<self.use_target_ft_ratio:
                    avg_ft = target_hidden_states[b][s:e].mean(0)
                else:
                    avg_ft = hidden_states[b][s:e].mean(0)
                    if self.emb_detach_ratio >0.0:
                        avg_ft = (1-self.emb_detach_ratio)*avg_ft + self.emb_detach_ratio*avg_ft.detach()
                with torch.no_grad():
                    trg_avg_ft = target_hidden_states[b][s:e].mean(0)
                averaged_target_hidden_states[b][s:e] = trg_avg_ft
                trg_avg_fts_.append(trg_avg_ft)
                avg_fts_.append(avg_ft)
            if len(avg_fts_)==0:
                continue
            trg_avg_fts.append(torch.stack(trg_avg_fts_))
            avg_fts.append(torch.stack(avg_fts_))
            avg_ft_attention_masks.append(torch.ones(len(trg_avg_fts_), device=hidden_states.device))

        distillation_loss = ((hidden_states-averaged_target_hidden_states)**2).sum(-1).mean()
        
        outputs = {'distillation_loss': distillation_loss,}


        if output_states:
            outputs['avg_fts'] =avg_fts
            outputs['avg_ft_attention_masks'] = avg_ft_attention_masks
            outputs['speech_states'] = hidden_states
            outputs['teacher_speech_states'] = averaged_target_hidden_states
            outputs['lm_logits']=lm_outputs
            outputs['teacher_lm_logits'] = teacher_lm_outputs
            outputs['segments'] = segments

        return outputs

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
