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
from torchaudio.models.rnnt import RNNT
from torchaudio.models.rnnt_decoder import RNNTBeamSearch
from torchaudio.transforms import RNNTLoss
from torchaudio.models.rnnt import _Predictor
from utils.evaluate import get_errors

class Joiner(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=None, identity_logit=False):
        super().__init__()
        
        hidden_dim = input_dim if hidden_dim is None else hidden_dim

        inter_dim = hidden_dim if not identity_logit else output_dim
        self.input_model = nn.Sequential(
                                nn.LayerNorm(input_dim),
                                nn.Linear(input_dim, hidden_dim),
                                nn.GELU(),
                                nn.Linear(hidden_dim, inter_dim),)
        if not identity_logit:
            self.logit = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                   nn.GELU(),
                                   nn.Linear(hidden_dim, output_dim),)
        else:
            self.logit = nn.Identity()

    def forward(
        self,
        source_encodings,
        source_lengths,
        target_encodings,
        target_lengths,
    ):
        
        source_encodings = self.input_model(source_encodings)
        joint_encodings = source_encodings.unsqueeze(2).contiguous() + target_encodings.unsqueeze(1).contiguous()
        output = self.logit(joint_encodings)
        return output, source_lengths, target_lengths
    
    
class SSLASR(nn.Module):

    def __init__(self,
                 joiner_configs,
                 predictor_configs,
                 transcriber_configs=None,
                 speech_upstream="facebook/hubert-base-ls960",
                 encoding_layer = 9,
                 load_hubert=False,
                 do_noise_augment=False,
                 use_proto_mixer=True,
                 noise_mixer_configs={},
                 segment_online = True,
                 thresholder_configs={},
                 merge_threshold_range=[0.8],
                 tokenizer_type="PhonemeTokenizer",
                 tokenizer_configs={},
                 beam_width=5,
                 step_max_tokens=100,
                 use_internal=False,
                 use_lm=False,
                 lm_layer=12,
                 use_ln_logit=True,
                 lm_kwargs={},
                 **kwargs,
                ):
        super().__init__()
        if load_hubert:
            self.speech_model = HubertModel.from_pretrained(speech_upstream, num_hidden_layers=encoding_layer)
        else:
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
        self.segment_online = segment_online
        if segment_online:
            self.thresholder = Thresholder(**thresholder_configs)
        else:
            self.thresholder = None
        self.merge_threshold_range=merge_threshold_range

        self.joiner = Joiner(**joiner_configs)
        self.predictor = _Predictor(**predictor_configs)
        self.tokenizer =  getattr(rnnt_tokenizer, tokenizer_type)(**tokenizer_configs)
        
        if transcriber_configs is not None:
            self.transcriber = nn.LSTM(bias=True, 
                                       batch_first=True,  bidirectional=True,
                                      **transcriber_configs)
            self.has_transcriber = True
        else:
            self.transcriber = nn.Identity()
            self.has_transcriber = False

        self.rnnt = RNNT(self.transcriber, 
                         self.predictor, 
                         self.joiner)
        self.loss=RNNTLoss(reduction='sum')
        self.token_decoder = self.tokenizer.get_decoder()
        self.remove_tokens = self.tokenizer.get_remove_tokens()
        self.blank_id=predictor_configs['num_symbols']-1
        self.search = RNNTBeamSearch(self.rnnt, self.blank_id,step_max_tokens=step_max_tokens)
        self.beam_width = beam_width
        
        self.use_internal = use_internal
        if use_internal:
            self.layer_weight = nn.Parameter(torch.ones(self.encoding_layer+1)/(self.encoding_layer+1), requires_grad=True)
        else:
            
            self.layer_weight = None
        self.no_segment = load_hubert
        
    def predict(self, input_values, attention_mask=None, beam_width=5, **kwargs):
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
            
        source_lengths = torch.LongTensor([len(segment) for segment in valid_segments]).to(source_encodings.device)
        if self.transcriber is not None:
            source_encodings = self.transcriber(source_encodings)
        transcribed_outputs=source_encodings
        with torch.no_grad():
            decoded=[self.search._search(trans[None,:l], None, beam_width=5, **kwargs)[0][0] for trans,l in zip(transcribed_outputs, source_lengths)]
        pred = [self.token_decoder([t for t in dec if t not in self.remove_tokens+[self.blank_id]])
                     for dec in decoded]

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
        if not self.no_segment:
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
                
            source_lengths = torch.LongTensor([len(segment) for segment in valid_segments]).to(source_encodings.device)
            if self.has_transcriber:
                source_encodings = self.transcriber(source_encodings)[0]
        else:
            source_encodings = features
            source_lengths = (attention_mask.sum(-1)//320).long()
            valid_texts = labels

        targets, target_lengths, texts = self.tokenizer(texts=valid_texts)
        targets = targets.to(source_encodings.device).to(torch.int32)
        target_encodings, target_lengths, predictor_state = self.predictor(
            input=targets,
            lengths=target_lengths,
            state=None,
        )
        output, source_lengths, target_lengths = self.joiner(
            source_encodings=source_encodings,
            source_lengths=source_lengths,
            target_encodings=target_encodings,
            target_lengths=target_lengths,
        )
        target_lengths = target_lengths.to(source_encodings.device).to(torch.int32)
        source_lengths = source_lengths.to(torch.int32).clip(0,output.shape[1])
        rnnt_loss =self.loss(logits=output,targets=targets[:,1:].contiguous(),
                        logit_lengths=source_lengths,
                        target_lengths=target_lengths-1,
                       ).mean() 

        if do_pred:
            transcribed_outputs=source_encodings
            with torch.no_grad():
                decoded=[self.search._search(trans[None,:l], None, beam_width=self.beam_width)[0][0] for trans,l in zip(transcribed_outputs, source_lengths)]
            pred = [self.token_decoder([t for t in dec if t not in self.remove_tokens+[self.blank_id]])
                         for dec in decoded]
        else:
            pred = []
        
                 
        outputs = {'rnnt_loss':rnnt_loss,
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
