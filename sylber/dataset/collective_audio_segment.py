import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
from pathlib import Path
import torchaudio
import torchaudio.functional as F
from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

        
class SpeechDataset(Dataset):
    
    def __init__(self, wav_dirs, tags, data_dir=None,  max_len=80000, dummy_len=300000, sample_by_ratio=True, noise_dir=None):
        super().__init__()
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.wav_dirs = [Path(wav_dir) for wav_dir in wav_dirs]
        if len(tags) == 1:
            sample_by_ratio=False
            tags = tags[0][1]
        self.sample_by_ratio = sample_by_ratio
        if sample_by_ratio:
            self.ratios = np.array([r for r,_ in tags])
            self.ratios = self.ratios/self.ratios.sum()
            self.tags = [ts for r,ts in tags]
        else:
            self.ratios = None
            self.tags = tags
        
        if sample_by_ratio:
            self.dummy_len = dummy_len
        else:
            self.dummy_len = len(self.tags)
        self.max_len = max_len
        if noise_dir is not None:
            self.noise_files = [f for f in Path(noise_dir).glob("*.wav")] + [f for f in Path(noise_dir).glob("*.flac")]
        else:
            self.noise_files = None
        
    def __len__(self):
        if self.sample_by_ratio:
            return self.dummy_len 
        else:
            return len(self.tags)

    def sample(self):
        di = np.random.choice(np.arange(len(self.ratios)), p=self.ratios)
        data = self.tags[di]
        tag = data[int(np.floor(np.random.uniform()*len(data)))]
        return tag, self.wav_dirs[di]
                              
    
    def __getitem__(self,i):
        if self.sample_by_ratio:
            tag, wav_dir = self.sample()
        else:
            tag, wav_dir = self.tags[i], self.wav_dirs[0]

        
        wav_file = wav_dir/f"{tag}.wav"
        if not wav_file.exists():
            wav_file = wav_dir/f"{tag}.flac"
        if not wav_file.exists():
            wav_file = wav_dir/f"{tag}.ogg"
        assert wav_file.exists()
        wav,sr = torchaudio.load(wav_file)
        if sr != 16000:
            wav = F.resample(wav, sr, 16000)
            sr = 16000
        wav = wav[0]
        buffer_size = 160
        frame_size = 320
        frame_len = len(wav)//frame_size
        max_frame_len = self.max_len//frame_size
        wav = wav[:frame_len*frame_size]
        if frame_len > max_frame_len:
            offset = np.random.randint(frame_len-max_frame_len)
            len_ = len(wav)
            wav = wav[offset*320:offset*320+self.max_len]
            s = offset
            e = offset+max_frame_len
        else:
            s = 0
            e = max_frame_len
        wav = torch.cat([torch.zeros(buffer_size, dtype=wav.dtype), wav, torch.zeros(buffer_size, dtype=wav.dtype)])

        if self.data_dir is not None:
            segments = np.load(self.data_dir/f"{tag}.npy")
            sampled_segs =[]
            for s_,e_ in segments:
                inter = min(e_,e)-max(s_,s)
                if inter>0:
                    sampled_segs.append([s_-s,e_-s])
            sampled_segs = np.array(sampled_segs).clip(0,max_frame_len)
        else:
            sampled_segs = None
        wav = wav.numpy()


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
            
        return {'wav': wav, 'segments': sampled_segs, 'tag':tag, 'range':[s,e],
               'noise':noise, }
    
    @staticmethod
    def collate(batch):
        data = {}
        processed = processor([d['wav'] for d in batch],
                              sampling_rate=16000, return_tensors="pt",
                              padding=True,return_attention_mask=True)
        data['input_values'] = processed.input_values
        data['attention_mask'] = processed.attention_mask
        if batch[0]['segments'] is None:
            data['segments']=None
        else:
            data['segments'] = [d['segments'] for d in batch]
        if batch[0]['noise'] is not None:
            data['noise'] = processor([d['noise'] for d in batch],
                              sampling_rate=16000, return_tensors="pt",
                              padding=True).input_values
        else:
            data['noise'] = None
        data['tag'] = [d['tag'] for d in batch]
        data['range'] = [d['range'] for d in batch]
        return data

class SpeechDataModule(LightningDataModule):
    def __init__(self,
                 data_dir,
                 train_files,
                 val_files,
                 test_files,
                 wav_dirs,
                 noise_dir=None,
                 max_len=80000,
                 dummy_len=300000,
                 batch_size=64,
                 val_batch_size=None,
                 num_workers=4,
                 drop_last=True,
                 pin_memory=True,
                 **kwargs,
                 ):
        super().__init__()
        
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        self.batch_size=batch_size
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.val_batch_size = batch_size if val_batch_size is None else val_batch_size
        self.max_len=max_len
        self.wav_dirs = wav_dirs
        self.dummy_len = dummy_len
        self.noise_dir = noise_dir
        
        
    def _load_tags(self, split):
        files = {"train": self.train_files,"valid":self.val_files,"test":self.test_files}[split]
        data = []
        for ratio, file in files:
            with open(file, 'r') as f:
                tags = [t.rstrip() for t in f.readlines()]
            data.append([ratio, tags])
        return data

                
    def train_dataloader(self):
        tags = self._load_tags('train')
        dataset = SpeechDataset(self.wav_dirs, tags,self.data_dir, max_len=self.max_len,
                               dummy_len=self.dummy_len, noise_dir=self.noise_dir)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=SpeechDataset.collate
        )
        return loader
    
    def val_dataloader(self):
        tags = self._load_tags('valid')
        dataset = SpeechDataset(self.wav_dirs, tags,self.data_dir, max_len=self.max_len,
                               dummy_len=self.dummy_len, noise_dir=self.noise_dir)
        loader = DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=SpeechDataset.collate
        )
        return loader
    
    def test_dataloader(self):
        tags = self._load_tags('test')
        dataset = SpeechDataset(self.wav_dirs, tags, self.data_dir, max_len=self.max_len,
                               dummy_len=self.dummy_len, noise_dir=self.noise_dir)
        loader = DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=SpeechDataset.collate
        )
        return loader
    