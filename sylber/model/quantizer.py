import torch
import torch.nn as nn
from vector_quantize_pytorch import GroupedResidualVQ
import numpy as np

def FeedForward(dim, dropout = 0.):
    dim_inner = dim
    return nn.Sequential(
        nn.Linear(dim, dim_inner),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim)
    )

class FFEncoder(nn.Module):
    def __init__(self,input_dim, output_dim, hidden_dims, dropout=0.0,):
        super().__init__()
        self.input_dim = input_dim
        modules = []
        
        for dim in hidden_dims:
            modules.append(nn.Linear(input_dim,dim))
            modules.append(FeedForward(dim, dropout=dropout))
            input_dim = dim
        
        modules.append(nn.Linear(input_dim,output_dim))
        
        self.mlp = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.mlp(x)

def _unit_norm(x):
    norm = (((x**2).sum(-1)+1e-5)**.5)[...,None]
    norm[norm==0] = 1.0
    x = x/norm
    return x

def _unit_norm_sep(x, spnorm, offset):
    if spnorm:
        x = torch.cat([_unit_norm(x[...,:-offset]), _unit_norm(x[...,-offset:])],-1)
        return x
    else:
        return _unit_norm(x)


def load_quantizer(config=None, ckpt=None):
    if config !=None:
        if not isinstance(config, dict):
            if not isinstance(config, dict):
                if config[-5:] == '.ckpt':
                    return load_quantizer(config=None, ckpt=config)
                else:
                    import yaml
                    with open(config) as f:
                        config = yaml.load(f, Loader=yaml.Loader)
        if 'model' in config.keys():
            config = config['model']
        state_dict = None
    else:
        assert ckpt != None
        ckpt = torch.load(ckpt)
        config = ckpt['config']
        state_dict = ckpt['state_dict']
        
    quantizer = Quantizer(**config)
    if ckpt != None and state_dict==None:
        ckpt = torch.load(ckpt)
        if 'state_dict' in ckpt.keys():
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
            
    if state_dict != None:
        quantizer.load_state_dict(state_dict, strict=True)
    quantizer = quantizer.eval()
    return quantizer

def load_km_quantizer(centroids, normalize=False):
    return KMQuantizer(centroids, normalize=normalize).eval()

def load_residualkm_quantizer(centroids, centroids2, normalize=False):
    return ResidualKMQuantizer(centroids, centroids2).eval()


class KMQuantizer(nn.Module):
    def __init__(self, centroids,normalize=False ):
        super().__init__()
        centroids = np.load(centroids)
        
        self.vq = GroupedResidualVQ(dim=768,
                                   num_quantizers = 1,
                                   codebook_size=centroids.shape[-1],
                                   decay=1.0,
                                    quantize_dropout=False,
                                    kmeans_init=False
                                   ) 
        device = self.vq.rvqs[0].layers[0]._codebook.embed
        self.vq.rvqs[0].layers[0]._codebook.embed = torch.from_numpy(centroids).to(device)[None,...]
        self.vq.rvqs[0].layers[0]._codebook.initted = torch.Tensor([True])
        self.normalize = normalize
    

    def get_indices(self, token):
        was_training = self.training
        self = self.eval()
        with torch.no_grad():
            if self.normalize:
                token =token/(((token**2).sum(-1)+1e-8)**.5)[...,None]*6
            outputs = self(token)
        
        if was_training:
            self = self.train()
            
        return outputs['indices']
            
    def _get_device(self):
        return self.encoder.mlp[0].weight.device
        
    def forward(self, token):
        # token: (B, L, d)
        quantized, indices, vq_loss = self.vq(token)
        
        outputs = {"indices":indices[0], "quantize":quantized, "non_quantized":token,
                   "commitment_loss": vq_loss}

        return outputs

    def decode(self, indices):
        # indices: (B, L, # art codebooks + # pitch codebooks)
        indices = indices.clip(0)
        indices = indices[...,:1]
        
        quantized = self.vq.get_output_from_indices(indices.unsqueeze(0))
        return quantized

class ResidualKMQuantizer(nn.Module):
    def __init__(self, centroids,centroids2,normalize=False ):
        super().__init__()
        self.km = KMQuantizer(centroids)
        self.km2 = KMQuantizer(centroids2)
    

    def get_indices(self, token):
        was_training = self.training
        self = self.eval()
        with torch.no_grad():
            idxs1 = self.km.get_indices(token)
            z_q = self.km.decode(idxs1)
            idxs2 = self.km2.get_indices(token-z_q)
            #outputs = self(token)
        
        if was_training:
            self = self.train()
        return torch.cat([idxs1, idxs2],-1)
            
    def _get_device(self):
        return self.encoder.mlp[0].weight.device
        
    def forward(self, token):
        # token: (B, L, d)
        
        quantized, indices, vq_loss = self.vq(token)
        
        outputs = {"indices":indices[0], "quantize":quantized, "non_quantized":token,
                   "commitment_loss": vq_loss}

        return outputs

    def decode(self, indices):
        # indices: (B, L, # art codebooks + # pitch codebooks)
        z_q1 = self.km.decode(indices[...,:1])
        z_q2 = self.km2.decode(indices[...,1:])
        quantized = z_q1+z_q2
        return quantized
        

        



class Quantizer(nn.Module):
    def __init__(self, encoder_configs, art_vq_configs, pitch_vq_configs,
                 unit_norm_encoder_input=True, unit_norm_encoder_output=True,
                 keep_blank_zero=True, pitch_emb_dim=8, separate_norm=True):
        super().__init__()
        self.encoder = FFEncoder(**encoder_configs)
        self.art_vq = GroupedResidualVQ(**art_vq_configs) 
        self.pitch_vq = GroupedResidualVQ(**pitch_vq_configs) 
        self.pitch_emb_dim = pitch_emb_dim
        self.art_emb_dim = encoder_configs['output_dim'] - self.pitch_emb_dim
        self.keep_blank_zero = keep_blank_zero
        self.separate_norm = separate_norm
        self.unit_norm_encoder_input = unit_norm_encoder_input
        self.unit_norm_encoder_output = unit_norm_encoder_output
        self.art_codebook_num = art_vq_configs["num_quantizers"]
        self.pitch_codebook_num = pitch_vq_configs["num_quantizers"]

    def get_indices(self, token):
        was_training = self.training
        self = self.eval()
        with torch.no_grad():
            outputs = self(token)

        if was_training:
            self = self.train()
            
        return outputs['indices']
            
    def _get_device(self):
        return self.encoder.mlp[0].weight.device
        
    def forward(self, token):
        # token: (B, L, d)
        
        non_blank_mask = (token**2).sum(-1)>0
        
        if self.unit_norm_encoder_input:
            token = _unit_norm(token)
           
        token = self.encoder(token)
        if self.unit_norm_encoder_output:
            token =_unit_norm_sep(token, self.separate_norm, self.pitch_emb_dim)
            
        if self.keep_blank_zero:
            token[~non_blank_mask] = 0.0
    
        art_token, pitch_token = token[...,:-self.pitch_emb_dim],  token[...,-self.pitch_emb_dim:]
        art_quantized, art_indices, art_vq_loss = self.art_vq(art_token)
        pitch_quantized, pitch_indices, pitch_vq_loss = self.pitch_vq(pitch_token)
        
        quantized = torch.cat([art_quantized, pitch_quantized],-1)
        
        if self.unit_norm_encoder_output:
            quantized = _unit_norm_sep(quantized, self.separate_norm, self.pitch_emb_dim)

        indices = torch.cat([art_indices[0], pitch_indices[0]], -1)  # (B, L, # art codebooks + # pitch codebooks)
        outputs = {"indices":indices, "quantize":quantized, "non_quantized":token,
                   "commitment_loss": art_vq_loss+pitch_vq_loss}

        return outputs

    def decode(self, indices):
        # indices: (B, L, # art codebooks + # pitch codebooks)
        indices = indices.clip(0)
        art_indices = indices[...,:self.art_codebook_num]
        pitch_indices = indices[...,self.art_codebook_num:]
        
        art_quantized = self.art_vq.get_output_from_indices(art_indices.unsqueeze(0))
        pitch_quantized = self.pitch_vq.get_output_from_indices(pitch_indices.unsqueeze(0))
        
        quantized = torch.cat([art_quantized, pitch_quantized],-1)
        
        if self.unit_norm_encoder_output:
            quantized = _unit_norm_sep(quantized, self.separate_norm, self.pitch_emb_dim)

        return quantized

        
