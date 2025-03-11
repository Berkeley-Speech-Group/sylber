import numpy as np
import torch
import torch.nn as nn
import torchaudio.functional as F
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
from pathlib import Path
import tqdm
import random
import csv
import soundfile as sf
import string
import re
import torchaudio
#import sys
#sys.append('..')
#from asr.tokenizer import BPETokenizer

from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
#tokenizer =  BPETokenizer()

class SpeechTextDataset(Dataset):
    
    def __init__(self, data, noise_dir=None, segment_dir=None):
        super().__init__()
        self.data = data
        if noise_dir is not None:
            self.noise_files = [f for f in Path(noise_dir).glob("*.wav")] + [f for f in Path(noise_dir).glob("*.flac")]
        else:
            self.noise_files = None
        
        self.segment_dir = Path(segment_dir) if segment_dir is not None else None
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,i):
        flac_path, text = self.data[i]
        wav,sr = torchaudio.load(flac_path)
        assert sr ==16000
        if self.segment_dir is not None:
            tag = Path(flac_path).stem
            segments = np.load(self.segment_dir/f"{tag}.npy")
        else:
            segments = None
        
        if not (self.noise_files is None):
            noise_file = self.noise_files[int(np.floor(np.random.uniform()*len(self.noise_files)))]
            noise,nsr = torchaudio.load(noise_file)

            if nsr != 16000:
                noise = F.resample(noise, nsr, 16000)
                nsr = 16000
            noise = noise[0]
            if len(noise)>len(wav):
                p = int(np.floor(np.random.uniform()*(len(noise)-len(wav))))
                noise = noise[p:p+len(wav)]
            wp =  int(np.floor(np.random.uniform()*(len(wav)-len(noise))).clip(0))
            noise_ = np.zeros_like(wav)
            noise_[wp:wp+len(noise)]=noise.numpy()
            noise = noise_
        else:
            noise = None

        
        output = {'wav':wav[0],
                  'text': text,
                 'noise':noise,
                 'segments':segments}
        
        return output
    
    @staticmethod
    def collate(batch):
        data = {}

        data['labels'] =[d['text'] for d in batch]
        '''
        decoder_attention_mask = torch.zeros_like(token_idxs)
        for bi, l in enumerate(token_lengths):
            decoder_attention_mask[bi,:l]=1
        
        data['decoder_attention_mask'] = decoder_attention_mask
        '''
        #wav_input = processor([d['wav'] for d in batch],
        #                           sampling_rate=16000, return_tensors="pt",
        #                           padding=True,return_attention_mask=True)
        if batch[0]['noise'] is not None:
            data['noise'] = processor([d['noise'] for d in batch],
                              sampling_rate=16000, return_tensors="pt",
                              padding=True).input_values
        else:
            data['noise'] = None
        wav_mean = np.mean([d['wav'].mean() for d in batch])
        wav_std = np.mean([d['wav'].std() for d in batch])
        input_values =  nn.utils.rnn.pad_sequence([d['wav'] for d in batch],
                                                batch_first=True, padding_value=0.0)
        input_values = (input_values-wav_mean)/wav_std
        data['input_values'] = input_values 
        data['attention_mask'] = nn.utils.rnn.pad_sequence([torch.ones(len(d['wav'])) for d in batch],
                                                batch_first=True, padding_value=0)
        if batch[0]['segments'] is None:
            data['segments']=None
        else:
            data['segments'] = [d['segments'] for d in batch]
        return data
    
                  
        

class SpeechTextDataModule(LightningDataModule):
    
    def __init__(self,
                 root_dir,
                 transcription='transcription',
                 max_len=20,
                 label_sr=None,
                 batch_size=64,
                 val_batch_size=None,
                 num_workers=4,
                 drop_last=True,
                 pin_memory=True,
                 segment_dir=None,
                 ):
        super().__init__()
        
        
        self.root_dir = Path(root_dir)
        self.transcription = transcription
        self.batch_size=batch_size
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.val_batch_size = batch_size if val_batch_size is None else val_batch_size
        self.max_len = max_len*16000
        self.segment_dir = segment_dir
        
    def _load_data(self, split):
        split_names={'train':  ['train-clean-100'], #, 'train-clean-360', 'train-other-500'],
                    'valid':['dev-clean'],
                    'test':['test-clean']}[split]
        
        data = []
        for split_name in split_names:
            texts=[]
            with open(str(self.root_dir/self.transcription/f'{split_name}.transcription.txt'), 'r') as f:
                texts = f.readlines()
            texts = [text.rstrip() for text in texts]

            tags=[]
            with open(str(self.root_dir/self.transcription/f'{split_name}.tag.txt'), 'r') as f:
                tags = f.readlines()
            tags = [tag.rstrip() for tag in tags]
            #regex = re.compile('[%s]' % re.escape('!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~'))
            for tag,text in zip(tags, texts):
                len_, tag=tag.split(' ')
                if int(len_)>self.max_len:
                    continue
                #text = regex.sub('', text.lower())
                data.append([str(self.root_dir/tag)+'.flac', text])
        return data
    
    
    
    def train_dataloader(self) -> DataLoader:
        
        data = self._load_data('train')
        dataset = SpeechTextDataset(data, segment_dir=self.segment_dir)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=SpeechTextDataset.collate
        )
        return loader
    
    def val_dataloader(self) -> DataLoader:
        
        data = self._load_data('valid')
        dataset = SpeechTextDataset(data, segment_dir=self.segment_dir)
        loader = DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=SpeechTextDataset.collate
        )
        return loader
    
    def test_dataloader(self) -> DataLoader:
        
        data = self._load_data('test')
        dataset = SpeechTextDataset(data, segment_dir=self.segment_dir)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=self.pin_memory,
            collate_fn=SpeechTextDataset.collate
        )
        return loader
    
