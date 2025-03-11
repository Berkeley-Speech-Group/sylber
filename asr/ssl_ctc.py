import math, random, string, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from lightning import LightningModule
except:
    lightning = None
from transformers import HubertModel, HubertConfig, BertModel, BertConfig
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel
#from transformers import GenerationConfig
from torch.optim.lr_scheduler import LambdaLR
from . import rnnt_tokenizer
from utils.segment_utils import cossim, get_segment, Thresholder
from utils.lr_schedule import LRLAMBDA, COSLRLAMBDA
from utils.noise_utils import NoiseMixerProto, NoiseMixer
import torchaudio
from utils.evaluate import get_errors
#from torchaudio.models.decoder import ctc_decoder

def get_sparsity_loss(x,dim=2):
    # x: (B, L, M, d)
    
    d = x.shape[dim]
    d_sq = x.shape[dim]**.5
    l2=(x**2).sum(dim=dim)**.5
    l1 = (((x-1/d)**2)**.5).sum(dim=dim)
    
    #loss = -(d_sq-l1/(l2+1e-5))/(d_sq-1)
    loss = l1/(l2+1e-5)
    return loss.mean()
    
class RFF(nn.Module):
    def __init__(self, dim, dropout=0.05,use_sn=False,use_bn=False,):
        super().__init__()
        if use_sn:
            self.linear1 = nn.utils.parametrizations.spectral_norm(nn.Linear(dim, dim))
            self.linear2 = nn.utils.parametrizations.spectral_norm(nn.Linear(dim, dim))
        else:
            self.linear1 = nn.Linear(dim, dim)
            self.linear2 = nn.Linear(dim, dim)
        self.dropout1= nn.Dropout(dropout)
        self.dropout2= nn.Dropout(dropout)
        if use_bn:
            self.bn1 = nn.BatchNorm1d(dim)
            self.bn2 = nn.BatchNorm1d(dim)
        else:
            self.bn1 = nn.LayerNorm(dim)
            self.bn2 = nn.LayerNorm(dim)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        orig_x = x
        x = self.bn1(x)
        x = self.activation(x)
        x = self.linear1(self.dropout1(x))
        x = self.bn2(x)
        x = self.activation(x)
        x = self.linear2(self.dropout2(x))
        x = x + orig_x
        return x

    def forward_wo_dropout(self, x):
        
        orig_x = x
        x = bn1=self.bn1(x)
        x = self.activation(x)
        x = self.linear1(x)
        x = bn2=self.bn2(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = x + orig_x
        return bn1,bn2
        

class Expander(nn.Module):
    def __init__(self, input_dim, output_dim, expand_factor, hidden_dim=None, add_blank=False,blank_id=-1,use_simple=False):
        super().__init__()
        

        
        hidden_dim = input_dim if hidden_dim is None else hidden_dim
        if use_simple:
            self.models =nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(expand_factor)])
            self.logit = nn.Linear(hidden_dim, output_dim)
        else:
            self.models =nn.ModuleList([nn.Sequential(
                                    #nn.LayerNorm(hidden_dim),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.GELU(),
                                    nn.Dropout(0.01),
                                    #*[RFF(hidden_dim) for _ in range(hidden_layer_n)],
                                    nn.Linear(hidden_dim, output_dim)) for _ in range(expand_factor)])
            self.logit = None
        self.expand_factor = expand_factor
        self.add_blank = add_blank
        self.blank_id = blank_id
        if add_blank:
            self.expand_factor += 1
            
    def forward(self, x):
        # x: (B, L, d)
        outputs = torch.cat([module(x).unsqueeze(2) for module in self.models],2) # B, L, M, d
        if self.add_blank:
            blanks = torch.ones_like(outputs[:,:,:1,:])*torch.finfo(outputs.dtype).min
            blanks[:,:,:,self.blank_id] = torch.finfo(outputs.dtype).max
            outputs = torch.cat([outputs,blanks],2)
        B,L,M,d = outputs.shape
        outputs = outputs.reshape(B, L*M, d)
        if self.logit is not None:
            outputs = self.logit(outputs)
        return outputs
        
        
    
    
class SSLASR(nn.Module):

    def __init__(self,
                 expander_configs,
                 speech_upstream="facebook/hubert-base-ls960",
                 encoding_layer = 9,
                 do_noise_augment=False,
                 use_proto_mixer=True,
                 noise_mixer_configs={},
                 segment_online = True,
                 thresholder_configs={},
                 merge_threshold_range=[0.8],
                 tokenizer_type="PhonemeTokenizer",
                 tokenizer_configs={},
                 use_internal=False,
                 use_lm=False,
                 lm_layer=12,
                 use_ln_logit=True,
                 lm_kwargs={},
                 sparsify=False,
                 transcriber_configs=None,
                 **kwargs,
                ):
        super().__init__()
        self.speech_model = HubertModel(HubertConfig.from_pretrained(speech_upstream, num_hidden_layers=encoding_layer))
        self.speech_model.requires_grad_(False)
        
        self.enc_dim = self.speech_model.config.hidden_size
        
        self.encoding_layer = encoding_layer

        if do_noise_augment:
            if use_proto_mixer:
                self.noise_mixer = NoiseMixerProto(**noise_mixer_configs)
            else:
                self.noise_mixer = NoiseMixer(**noise_mixer_configs)
        else:
            self.noise_mixer = None
            
        if use_lm:
            self.language_model = BertModel(BertConfig.from_pretrained("google-bert/bert-base-uncased",
                                                                       num_hidden_layers=lm_layer, **lm_kwargs))
            
            self.lm_enc_dim = self.language_model.config.hidden_size
            self.masked_embed = nn.Parameter(torch.Tensor(self.enc_dim), requires_grad=True)
            nn.init.normal_(self.masked_embed,std=1.0/math.sqrt(self.enc_dim))
            self.input_linear = nn.Linear(self.enc_dim, self.lm_enc_dim)
            if use_ln_logit:
                self.logit = nn.Sequential(nn.LayerNorm(self.lm_enc_dim),
                                    nn.Linear(self.lm_enc_dim,self.lm_enc_dim))
            else:
                
                self.logit = nn.Sequential(nn.Linear(self.lm_enc_dim, self.lm_enc_dim*2),
                                             nn.GELU(),
                                             nn.Linear(self.lm_enc_dim*2, self.lm_enc_dim)
                                            )
            self.language_model.requires_grad_(False)
            self.logit.requires_grad_(False)
            self.input_linear.requires_grad_(False)
            self.lm_layer_weight = nn.Parameter(torch.ones(lm_layer+1)/(lm_layer+1), requires_grad=True)
            use_internal = False
        else:
            self.language_model = None
            self.logit = nn.Identity()
        self.segment_online = segment_online
        if segment_online:
            self.thresholder = Thresholder(**thresholder_configs)
        else:
            self.thresholder = None
        self.merge_threshold_range=merge_threshold_range

        self.tokenizer =  getattr(rnnt_tokenizer, tokenizer_type)(**tokenizer_configs)
        
        self.token_decoder = self.tokenizer.get_decoder()
        self.remove_tokens = self.tokenizer.get_remove_tokens()
        self.blank_id=self.tokenizer.blank
        self.expander = Expander(blank_id=self.blank_id,
                                 **expander_configs)
        self.loss = nn.CTCLoss(blank = self.blank_id, zero_infinity=True)
        self.use_internal = use_internal
        if use_internal:
            self.layer_weight = nn.Parameter(torch.ones(self.encoding_layer+1)/(self.encoding_layer+1), requires_grad=True)
        else:
            
            self.layer_weight = None
        self.sparsify = sparsify
        if transcriber_configs is not None:
            self.transcriber = nn.LSTM(bias=True, 
                                       batch_first=True,  bidirectional=True,

                                       input_size=transcriber_configs["input_size"],
                                       
    hidden_size=transcriber_configs["hidden_size"],
    num_layers=transcriber_configs["num_layers"],
    dropout=transcriber_configs["dropout"])
            self.logit = nn.Linear(transcriber_configs['hidden_size']*2,transcriber_configs['output_dim'])
            self.has_transcriber = True
        else:
            self.transcriber = nn.Identity()
            self.has_transcriber = False
        '''
        self.decoder = ctc_decoder( lexicon=None,
                          tokens=self.tokenizer.vocab,
                          lm=None,
                          lm_dict=None,
                          nbest=1,
                          beam_size=20,
                          blank_token="BLANK",
                          sil_token="PAD", #str(model.tokenizers[-1].pad_id()),
                         )
        '''
    def segment(self, input_values, attention_mask=None, **kwargs):
        """
        """        

        with torch.no_grad():
            self.speech_model.eval()
            outputs = self.speech_model(input_values, attention_mask=attention_mask, output_hidden_states=self.use_internal)
            hidden_states = outputs.last_hidden_state
        if self.use_internal:
            weight = self.layer_weight[:,None,None,None]/self.layer_weight.sum()
            features = (torch.stack(outputs.hidden_states)*weight).sum(0)
        else:
            features = hidden_states
            

        assert self.segment_online
        normthreshold = self.thresholder.get_threshold().item()
        with torch.no_grad():
            norms = ((hidden_states**2).sum(-1)+1e-8)**.5
            if self.merge_threshold_range[0]<self.merge_threshold_range[1]:
                merge_threshold = np.random.uniform(*self.merge_threshold_range)
            else:
                merge_threshold = self.merge_threshold_range[0]
            segments=[get_segment(states.cpu().numpy(), normthreshold, merge_threshold) for states in hidden_states]

        return features, segments
        
    def predict(self, input_values, attention_mask=None, output_posterior=False, normalize=False, **kwargs):
        """
        """        

        with torch.no_grad():
            self.speech_model.eval()
            outputs = self.speech_model(input_values, attention_mask=attention_mask, output_hidden_states=self.use_internal)
            hidden_states = outputs.last_hidden_state
        if self.use_internal:
            weight = self.layer_weight[:,None,None,None]/self.layer_weight.sum()
            features = (torch.stack(outputs.hidden_states)*weight).sum(0)
        else:
            features = hidden_states
            

        assert self.segment_online
        normthreshold = self.thresholder.get_threshold().item()
        with torch.no_grad():
            norms = ((hidden_states**2).sum(-1)+1e-8)**.5
            if self.merge_threshold_range[0]<self.merge_threshold_range[1]:
                merge_threshold = np.random.uniform(*self.merge_threshold_range)
            else:
                merge_threshold = self.merge_threshold_range[0]
            segments=[get_segment(states.cpu().numpy(), normthreshold, merge_threshold) for states in hidden_states]

    
        valid_segments = [segment for segment in segments if len(segment) >0]
        avg_fts = []
        avg_ft_attention_masks = []
        for b in range(len(hidden_states)):
            avg_fts_ = []
            for s,e in segments[b]:
                avg_ft = features[b][s:e].mean(0)
                if normalize:
                    avg_ft = avg_ft/((avg_ft**2).sum(-1)+1e-8)**.5
                    avg_ft = avg_ft * self.thresholder.signal_mean.item()
                avg_fts_.append(avg_ft)
            if len(avg_fts_) != 0:
                avg_fts.append(torch.stack(avg_fts_))
            avg_ft_attention_masks.append(torch.ones(len(avg_fts_), device=features.device))
            
        source_encodings = nn.utils.rnn.pad_sequence(avg_fts, batch_first=True, padding_value=0.0)
        avg_ft_attention_masks = nn.utils.rnn.pad_sequence(avg_ft_attention_masks, batch_first=True, padding_value=0)   
        if self.language_model is not None:
            with torch.no_grad():
                lm_outputs = self.language_model(attention_mask=avg_ft_attention_masks, 
                                              inputs_embeds=self.input_linear(source_encodings),
                                                output_hidden_states=True).hidden_states
            weight = self.lm_layer_weight[:,None,None,None]/self.lm_layer_weight.sum()
            source_encodings = (torch.stack(lm_outputs)*weight).sum(0)

        source_lengths = torch.LongTensor([len(segment)*self.expander.expand_factor for segment in valid_segments]).to(source_encodings.device)
        source_encodings = self.expander(source_encodings)
        if self.transcriber is not None:
            source_encodings = self.logit(self.transcriber(source_encodings)[0])
        #source_encodings = source_encodings.log_softmax(-1)
        pred_idxs = source_encodings.argmax(-1)
        decoded = [torch.unique_consecutive(pred_idx[:source_length]).cpu().numpy() for pred_idx, source_length in zip(pred_idxs, source_lengths)]
        pred = [self.token_decoder([t for t in dec if t not in self.remove_tokens+[self.blank_id]])
                 for dec in decoded]

        
        if output_posterior:
            with torch.no_grad():
                posterior = source_encodings.softmax(-1)
            return pred,posterior
        else:
            return pred
        

    def forward(self, input_values, labels, segments=None,attention_mask=None, noise=None,do_pred=False, **kwargs):
        """
        """        

        if self.noise_mixer is not None:
            assert noise is not None
            with torch.no_grad():
                input_values = self.noise_mixer(input_values, noise)
                
        with torch.no_grad():
            self.speech_model.eval()
            outputs = self.speech_model(input_values, attention_mask=attention_mask, output_hidden_states=self.use_internal)
            hidden_states = outputs.last_hidden_state
        if self.use_internal:
            weight = self.layer_weight[:,None,None,None]/self.layer_weight.sum()
            features = (torch.stack(outputs.hidden_states)*weight).sum(0)
        else:
            features = hidden_states
            
        if segments is None:
            assert self.segment_online
            normthreshold = self.thresholder.get_threshold().item()
            with torch.no_grad():
                norms = ((hidden_states**2).sum(-1)+1e-8)**.5
                if self.merge_threshold_range[0]<self.merge_threshold_range[1]:
                    merge_threshold = np.random.uniform(*self.merge_threshold_range)
                else:
                    merge_threshold = self.merge_threshold_range[0]
                segments=[get_segment(states.cpu().numpy(), normthreshold, merge_threshold) for states in hidden_states]

        
        valid_segments = [segment for segment in segments if len(segment) >0]
        valid_texts = [text for bi,text in enumerate(labels) if len(segments[bi]) > 0]
        avg_fts = []
        avg_ft_attention_masks = []
        for b in range(len(hidden_states)):
            avg_fts_ = []
            for s,e in segments[b]:
                avg_ft = features[b][s:e].mean(0)
                avg_fts_.append(avg_ft)
            if len(avg_fts_) != 0:
                avg_fts.append(torch.stack(avg_fts_))
            avg_ft_attention_masks.append(torch.ones(len(avg_fts_), device=features.device))
            
        source_encodings = nn.utils.rnn.pad_sequence(avg_fts, batch_first=True, padding_value=0.0)
        avg_ft_attention_masks = nn.utils.rnn.pad_sequence(avg_ft_attention_masks, batch_first=True, padding_value=0)   
        if self.language_model is not None:
            with torch.no_grad():
                lm_outputs = self.language_model(attention_mask=avg_ft_attention_masks, 
                                              inputs_embeds=self.input_linear(source_encodings),
                                                output_hidden_states=True).hidden_states
            weight = self.lm_layer_weight[:,None,None,None]/self.lm_layer_weight.sum()
            source_encodings = (torch.stack(lm_outputs)*weight).sum(0)
        
        source_lengths = torch.LongTensor([len(segment)*self.expander.expand_factor for segment in valid_segments]).to(source_encodings.device)
        source_encodings = self.expander(source_encodings)
        if self.has_transcriber:
            source_encodings = self.logit(self.transcriber(source_encodings)[0])
        source_encodings = source_encodings.log_softmax(-1)

        targets, target_lengths, texts = self.tokenizer(texts=valid_texts)

        ## The tokenizer is reused from RNN-T one which has a blank token attached at the front.
        targets = targets[:,1:]
        target_lengths = target_lengths -1
        
        targets = targets.to(source_encodings.device).to(torch.int32)

        loss = self.loss(source_encodings.transpose(0,1), targets, source_lengths, target_lengths)

        if do_pred:
            with torch.no_grad():
                pred_idxs = source_encodings.argmax(-1)

                decoded = [torch.unique_consecutive(pred_idx[:source_length]).cpu().numpy() for pred_idx, source_length in zip(pred_idxs, source_lengths)]
            pred = [self.token_decoder([t for t in dec if t not in self.remove_tokens+[self.blank_id]])
                         for dec in decoded]
        else:
            pred = []
        if self.sparsify:
            probs = source_encodings.exp()
            B, LM, d = probs.shape
            probs = probs.reshape(B, LM//self.expander.expand_factor, self.expander.expand_factor, d)
            sparsity_loss = get_sparsity_loss(probs,dim=2)
        else:
            sparsity_loss = 0
                 
        outputs = {'ctc_loss':loss,
                   'sparsity_loss':sparsity_loss,
                  'pred':pred,
                  'texts':texts,}
        return outputs

class ASRTrainer(LightningModule):

    def __init__(self, loss_coefs,lr=0.0001, warmup_steps=5000, total_steps=500000, min_factor=0.05, use_cos_sch=False, 
                 **model_configs):
        super().__init__()
        self.loss_coefs = loss_coefs
        self.net = SSLASR(**model_configs).to(torch.float)
        self.lr = lr
        self.total_steps =total_steps
        self.warmup_steps = warmup_steps
        self.min_factor = min_factor
        self.use_cos_sch = use_cos_sch
        
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
        outputs = self.net(**batch, do_pred=True)
        loss_val = 0
        
        for coef_name, coef_val in self.loss_coefs.items():
            if coef_name in outputs.keys():
                loss_val += coef_val * outputs[coef_name]
                self.log(f'val_{coef_name}', outputs[coef_name],sync_dist=True)
        self.log(f'val_loss', loss_val,sync_dist=True)
        pred = outputs['pred']
        errors = get_errors( pred, outputs['texts'])
        
        for error_name, error in errors.items():
            self.log(f'val_{error_name}', error,sync_dist=True)
        #self.log(f'val_wer', outputs['wer'],sync_dist=True)
        return loss_val

    def configure_optimizers(self):
        
        opt_fun = torch.optim.AdamW
        opt = opt_fun(self.net.parameters(),lr=self.lr,eps=1e-4, betas=(0.9, 0.95), weight_decay = 0.1)
        # Create the scheduler
        if self.use_cos_sch:
            lr_lambda = COSLRLAMBDA(self.warmup_steps, self.total_steps, self.min_factor)
        else:
            lr_lambda = LRLAMBDA(self.warmup_steps, self.total_steps, self.min_factor)
        sch = LambdaLR(opt, lr_lambda)
        return [opt],[{"scheduler": sch, "interval": "step"}]
