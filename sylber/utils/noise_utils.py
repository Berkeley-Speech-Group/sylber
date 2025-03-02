import torch
import numpy as np

class NoiseMixer():
    def __init__(self, augment_prob=0.2, utterance_mix_ratio=0.25, shift_range=[0.4,0.7],
                 magnitude_range=[0.05,0.7],utterance_magnitude_max_scale=0.2):
        self.augment_prob = augment_prob
        self.utterance_mix_ratio = utterance_mix_ratio
        self.shift_range = shift_range
        self.magnitude_range = magnitude_range
        self.utterance_magnitude_max_scale=utterance_magnitude_max_scale

    def __call__(self, wav, noise):
        # wav: (B,L,)
        # noise: (B,L,)

        is_augmented = (torch.rand_like(wav[:,0])<= self.augment_prob)*1.0 # (B,)
        is_utterance_mix = (torch.rand_like(wav[:,0])<= self.utterance_mix_ratio)*1.0 # (B,)
        idxs = np.arange(len(wav))
        np.random.shuffle(idxs)
        shuffled_wav = wav[idxs]
        
        shift_ranges = torch.rand_like(wav[:,0])*(self.shift_range[1]-self.shift_range[0])+self.shift_range[0]
        left_shift_mask = (torch.linspace(0,1,noise.shape[1], device=noise.device)[None,:] > shift_ranges[:,None])*1.0
        right_shift_mask = (torch.linspace(1,0,noise.shape[1], device=noise.device)[None,:] > shift_ranges[:,None])*1.0
        is_left = (torch.rand_like(wav[:,0])>=0.5)*1.0

        is_utterance_mix = is_utterance_mix[:,None]
        is_left = is_left[:,None]
        
        noise = (1-is_utterance_mix)*noise + is_utterance_mix*(is_left*left_shift_mask*shuffled_wav+(1-is_left)*right_shift_mask*shuffled_wav)        
        magnitude = torch.rand_like(wav[:,0])*(self.magnitude_range[1]-self.magnitude_range[0])+self.magnitude_range[0]
        utt_magnitude = torch.rand_like(wav[:,0])*(self.utterance_magnitude_max_scale-self.magnitude_range[0])+self.magnitude_range[0]
        magnitude = utt_magnitude*is_utterance_mix.squeeze(-1) + (1-is_utterance_mix.squeeze(-1))*magnitude
        magnitude = is_augmented[:,None]*magnitude[:,None]
        wav = wav+magnitude*noise
        return wav
        