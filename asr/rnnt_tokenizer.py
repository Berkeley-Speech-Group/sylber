import torch.nn as nn
import torchaudio
import torch
import string
import numpy as np
try:
    import soundfile as sf
    import librosa
except:
    librosa = None
    sf = None


class GraphemeTokenizer(nn.Module):
    def __init__(self,include_space=True, include_apostrophe=True, **kwargs):
        super().__init__()
        self.charactors = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
                           'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
                           'u', 'v', 'w', 'x', 'y', 'z',]
        if include_space:
            self.charactors.append(' ')
        if include_apostrophe:
            self.charactors.append("'")
        self.ch2idx = {ch:i for i, ch in enumerate(self.charactors)}
        self.pad_id = len(self.charactors) 
        self.blank =  len(self.charactors)+1

        self.charactors += ["PAD", "BLANK"]
        self.vocab = self.charactors
        
        
    def __call__(self, texts, **kwargs):
        texts=[text.lower() for text in texts]
        token_idxs = [[self.ch2idx[t] for t in text] for text in texts]
        token_idxs = [torch.tensor([self.blank] + idxs) for idxs in token_idxs]
        token_lengths = torch.tensor([len(idxs) for idxs in token_idxs])
        
        token_idxs = nn.utils.rnn.pad_sequence(token_idxs,batch_first=True, padding_value=self.pad_id)
        return token_idxs, token_lengths, texts
    
    def get_decoder(self):
        return lambda tokens: ''.join([self.charactors[i] for i in tokens])
    
    def get_remove_tokens(self):
        return [self.pad_id, self.blank]
    
    def pad_id(self):
        return self.pad_id
    
    def bos_id(self):
        return self.blank
    
    def eos_id(self):
        return self.blank
    
class PhonemeTokenizer(nn.Module):
    def __init__(self, include_stress=True,include_space=False,wrap_sent=False, **kwargs):
        super().__init__()
        from g2p_en import G2p
        self.g2p = G2p()
        self.phonemes = ['SIL', 'SPN', 'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2',
                         'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH',
                         'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2',
                         'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L',
                         'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S',
                         'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
        
        self.include_space = include_space
        self.wrap_sent = wrap_sent
        
        if self.include_space:
            self.phonemes.append('|')
            
        
        self.include_stress = include_stress
        if not self.include_stress:
            phonemes_ = []
            for ph in self.phonemes:
                ph=self._remove_stressed_vowel(ph)
                if ph not in phonemes_:
                    phonemes_.append(ph)
            self.phonemes=phonemes_
        if '|' in self.phonemes:
            self.space_id = self.phonemes.index('|')
        if self.wrap_sent:
            self.phonemes += ['<BOS>', '<EOS>']
            self.bos_id = len(self.phonemes)-2
            self.eos_id = len(self.phonemes)-1
            
        self.ph2idx = {ph:i for i, ph in enumerate(self.phonemes)}
        self.pad_id = len(self.phonemes) 
        self.blank =  len(self.phonemes)+1
        self.vocab_size = len(self.phonemes)+2
        self.phonemes += ["PAD", "BLANK"]
        self.vocab = self.phonemes
        
    @staticmethod
    def _remove_stressed_vowel(ph):
        if ph[-1] in ['0','1','2']:
            ph = ph[:-1]
        return ph
    
    @staticmethod
    def _convert_space(ph):
        return '|' if ph ==' ' else ph
    
    def __call__(self, texts, **kwargs):
        texts = [self.g2p(text) for text in texts]
        if not self.include_stress:
            texts = [[self._remove_stressed_vowel(ph) for ph in text] for text in texts]
        texts = [[self._convert_space(ph) for ph in text] for text in texts]
        texts = [[ph for ph in text if ph in self.phonemes] for text in texts]
        token_idxs = [[self.ph2idx[t.upper()] for t in text] for text in texts]
        if self.wrap_sent:
            token_idxs = [[self.bos_id]+idxs+[self.eos_id] for idxs in token_idxs]
        token_idxs = [torch.tensor([self.blank] + idxs) for idxs in token_idxs]
        token_lengths = torch.tensor([len(idxs) for idxs in token_idxs])
        
        token_idxs = nn.utils.rnn.pad_sequence(token_idxs,batch_first=True, padding_value=self.pad_id)
        return token_idxs, token_lengths, [' '.join(text) for text in texts]

    def get_decoder(self):
        return lambda tokens: ' '.join([self.phonemes[i] for i in tokens])
    
    def get_remove_tokens(self):
        remove_tokens = [self.pad_id, self.blank, 0, 1]
        if self.wrap_sent:
            remove_tokens+=[self.bos_id, self.eos_id, 0, 1]
        return remove_tokens
    
    def pad_id(self):
        return self.pad_id
    
    def bos_id(self):
        return self.blank
    
    def eos_id(self):
        return self.blank


class TIMITPhonemeTokenizer(nn.Module):
    def __init__(self, include_stress=True,include_space=False,wrap_sent=False, **kwargs):
        super().__init__()
        self.phonemes = ['aa',
                         'ae',
                         'ah',
                         'ao',
                         'aw',
                         'ax',
                         'ax-h',
                         'axr',
                         'ay',
                         'b',
                         'bcl',
                         'ch',
                         'd',
                         'dcl',
                         'dh',
                         'dx',
                         'eh',
                         'el',
                         'em',
                         'en',
                         'eng',
                         'epi',
                         'er',
                         'ey',
                         'f',
                         'g',
                         'gcl',
                         'h#',
                         'hh',
                         'hv',
                         'ih',
                         'ix',
                         'iy',
                         'jh',
                         'k',
                         'kcl',
                         'l',
                         'm',
                         'n',
                         'ng',
                         'nx',
                         'ow',
                         'oy',
                         'p',
                         'pau',
                         'pcl',
                         'q',
                         'r',
                         's',
                         'sh',
                         't',
                         'tcl',
                         'th',
                         'uh',
                         'uw',
                         'ux',
                         'v',
                         'w',
                         'y',
                         'z',
                         'zh']
        
        self.include_space = include_space
        self.wrap_sent = wrap_sent
        
        if self.include_space:
            self.phonemes.append('|')
            
        
        self.include_stress = include_stress
        if not self.include_stress:
            phonemes_ = []
            for ph in self.phonemes:
                ph=self._remove_stressed_vowel(ph)
                if ph not in phonemes_:
                    phonemes_.append(ph)
            self.phonemes=phonemes_
        if '|' in self.phonemes:
            self.space_id = self.phonemes.index('|')
        if self.wrap_sent:
            self.phonemes += ['<BOS>', '<EOS>']
            self.bos_id = len(self.phonemes)-2
            self.eos_id = len(self.phonemes)-1
            
        self.ph2idx = {ph:i for i, ph in enumerate(self.phonemes)}
        self.pad_id = len(self.phonemes) 
        self.blank =  len(self.phonemes)+1
        self.vocab_size = len(self.phonemes)+2
        
    @staticmethod
    def _remove_stressed_vowel(ph):
        if ph[-1] in ['0','1','2']:
            ph = ph[:-1]
        return ph
    
    @staticmethod
    def _convert_space(ph):
        return '|' if ph ==' ' else ph
    
    def __call__(self, texts, **kwargs):
        texts = [text.lower().split(' ') for text in texts]
        if not self.include_stress:
            texts = [[self._remove_stressed_vowel(ph) for ph in text] for text in texts]
        texts = [[self._convert_space(ph) for ph in text] for text in texts]
        texts = [[ph for ph in text if ph in self.phonemes] for text in texts]
        token_idxs = [[self.ph2idx[t] for t in text] for text in texts]
        if self.wrap_sent:
            token_idxs = [[self.bos_id]+idxs+[self.eos_id] for idxs in token_idxs]
        token_idxs = [torch.tensor([self.blank] + idxs) for idxs in token_idxs]
        token_lengths = torch.tensor([len(idxs) for idxs in token_idxs])
        
        token_idxs = nn.utils.rnn.pad_sequence(token_idxs,batch_first=True, padding_value=self.pad_id)
        return token_idxs, token_lengths, [' '.join(text) for text in texts]

    def get_decoder(self):
        return lambda tokens: ' '.join([self.phonemes[i] for i in tokens])
    
    def get_remove_tokens(self):
        remove_tokens = [self.pad_id, self.blank, 0, 1]
        if self.wrap_sent:
            remove_tokens+=[self.bos_id, self.eos_id, 0, 1]
        return remove_tokens
    
    def pad_id(self):
        return self.pad_id
    
    def bos_id(self):
        return self.blank
    
    def eos_id(self):
        return self.blank
    
class SyllableTokenizer(nn.Module):
    def __init__(self, syllable_vocab_file=None, **kwargs):
        super().__init__()
        if syllable_vocab_file is None:
            syllable_vocab_file = Path(__file__).parent.parent/'misc'/'syllables.txt'
        with open(syllable_vocab_file, 'r') as f:
            syllables=f.readlines()
        self.syllables = [syl.rstrip() for syl in syllables]
        self.syllables.append('<UNK>')
        self.syl2idx = {syl:i for i, syl in enumerate(self.syllables)}
        self.pad_id = len(self.syllables) 
        self.blank =  len(self.syllables)+1
        
    def _syl2idx(self, syl):
        return self.syl2idx[syl] if syl in self.syllables else self.syl2idx['<UNK>']
              
    def __call__(self, texts, **kwargs):
        token_idxs = [[self._syl2idx(t) for  t in text.split(' ')] for text in texts]
        token_idxs = [torch.tensor([self.blank] + idxs) for idxs in token_idxs]
        token_lengths = torch.tensor([len(idxs) for idxs in token_idxs])
        
        token_idxs = nn.utils.rnn.pad_sequence(token_idxs,batch_first=True, padding_value=self.pad_id)
        return token_idxs, token_lengths, texts
    
    def get_decoder(self):
        return lambda tokens: ' '.join([self.syllables[i] for i in tokens])
    
    def get_remove_tokens(self):
        return [self.pad_id, self.blank]
    
    def pad_id(self):
        return self.pad_id
    
    def bos_id(self):
        return self.blank
    
    def eos_id(self):
        return self.blank   

class BPETokenizer(nn.Module):
    
    def __init__(self,blank,**kwargs):
        super().__init__()
        self.blank=blank
        bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH
        self.tokenizer = bundle.get_token_processor()
        self.vocab_size = 4096+2
        
    def __call__(self, texts, **kwargs):
        texts=[text.lower() for text in texts]
        token_idxs = self.tokenizer.sp_model.Encode(texts)
        token_idxs = [torch.tensor([self.blank] + idxs) for idxs in token_idxs]
        token_lengths = torch.tensor([len(idxs) for idxs in token_idxs])
        
        token_idxs = nn.utils.rnn.pad_sequence(token_idxs,batch_first=True, padding_value=self.tokenizer.sp_model.pad_id())
        return token_idxs, token_lengths, texts
    
    def get_decoder(self):
        return self.tokenizer.sp_model.DecodeIds
    
    def get_remove_tokens(self):
        return [self.tokenizer.sp_model.pad_id(),
                self.tokenizer.sp_model.bos_id(),
                self.tokenizer.sp_model.eos_id()]
    
    def pad_id(self):
        return self.tokenizer.sp_model.pad_id()
    
    def bos_id(self):
        return self.tokenizer.sp_model.bos_id()
    
    def eos_id(self):
        return self.tokenizer.sp_model.eos_id()
    
class HuBERTTokenizer(nn.Module):
    
    def __init__(self, pre_tokenized=True, km_n=100, device='cuda', collapse=True, spm=None, **kwargs):
        super().__init__()
        if not pre_tokenized:
            assert km_n in [50, 100, 200, 500, 2000], "Only km_n in [50, 100, 200] is supported"
            from pathlib import Path
            module_path = Path(__file__)
            self.km_path = module_path.parent.parent/'km_models'
            if not self.km_path.exists():
                self.km_path.mkdir(exist_ok=True)
            self.layer = 6
            if km_n in [50, 100, 200]:
                self.layer = 6
            elif km_n in [500, 2000]:
                self.layer = 9
                
            self.km_path = self.km_path/f'hubert-l{self.layer}-km{km_n}.bin'
            if not self.km_path.exists():
                import wget
                if km_n in [50, 100, 200]:
                    _ = wget.download(f'https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km{km_n}/km.bin', out=str(self.km_path))
                else:
                    if km_n == 500:
                        _ = wget.download('https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin', out=str(self.km_path))
                    elif km_n == 2000:
                        _ = wget.download('https://dl.fbaipublicfiles.com/textless_nlp/expresso/checkpoints/hubert_base_ls960_L9_km2000_expresso.bin', out=str(self.km_path))
            import joblib
            self.km_model = joblib.load(open(str(self.km_path), "rb"))
            self.km_model.verbose = False
            from transformers import Wav2Vec2Processor, HubertModel
            self.device = device
            self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
            self.speech_encoder = HubertModel.from_pretrained('facebook/hubert-base-ls960').to(self.device)
            
        if spm is not None:
            import sentencepiece
            from pathlib import Path
            if not Path(spm).exists():
                spm = str(Path(__file__).parent.parent/'spm'/f'hubert-l6_km{km_n}_{spm}.model')
            
            spm =str(spm)
            self.spm = sentencepiece.SentencePieceProcessor(model_file=spm)
            self.valid_unicode = np.load(Path(__file__).parent.parent/'misc'/'valid_unicode.npy')
            self.reverse = {code:i for i,code in enumerate(self.valid_unicode)}
            spm_file_tag=Path(spm).stem
            if 'bpe' in spm_file_tag:
                vocab_size = int(spm_file_tag.split('bpe')[-1])
            elif 'unigram' in spm_file_tag:
                vocab_size = int(spm_file_tag.split('unigram')[-1])
            else:
                raise NotImplemented
            self.pad_id = vocab_size
            self.blank = vocab_size+1
            self.collapse=True
            self.vocab_size = vocab_size+2
        else:
            self.spm = None 
            self.pad_id = km_n 
            self.blank = km_n+1
            self.collapse = collapse
            self.vocab_size = km_n+2
            
        self.pre_tokenized = pre_tokenized
        
    def get_feature(self, wav, sr=16000):
        assert not self.pre_tokenized, "set pre_tokenized=False to load quantizer model"
        inputs=self.processor(wav,sampling_rate=sr, return_tensors="pt")
        inputs=inputs.to(self.device)
        with torch.no_grad():
            outputs = self.speech_encoder(**inputs,output_hidden_states=True)
        states=outputs.hidden_states[self.layer].squeeze(0).cpu().numpy()
        return states
    
    def tokenize(self, wav, sr=16000):
        assert not self.pre_tokenized, "set pre_tokenized=False to load quantizer model"
        inputs=self.processor(wav,sampling_rate=sr, return_tensors="pt")
        inputs=inputs.to(self.device)
        with torch.no_grad():
            outputs = self.speech_encoder(**inputs,output_hidden_states=True)
        states=outputs.hidden_states[self.layer].squeeze(0).cpu().numpy()
        tokens = self.km_model.predict(states)
        return tokens

    def segment(self, wav, sr=16000):
        tokens = self.tokenize(wav,sr)
        boundaries = list(np.nonzero(np.diff(tokens))[0]+1)
        boundaries = [0] + boundaries + [len(tokens)]
        boundaries = np.array(boundaries)
        segments = np.stack([boundaries[:-1],boundaries[1:]]).T
        segfts = tokens[segments[:,0]]
        return {'segments':segments, 'segment_features':segfts}
    
    @staticmethod
    def _collapse_tensor(tokens):
        is_continuous = tokens[1:]==tokens[:-1]
        return torch.cat([tokens[:1], tokens[1:][~is_continuous]])
    
    @staticmethod
    def _collapse_numpy(tokens):
        is_continuous = tokens[1:]==tokens[:-1]
        return np.concatenate([tokens[:1], tokens[1:][~is_continuous]])
    
    def _spm_decode(self, tokens):
        tokens=self.spm.DecodeIds(tokens)
        new_tokens=[]
        for t in tokens:
            t = ord(t)
            if t in self.reverse.keys():
                new_tokens.append(str(self.reverse[t]))
        decoded = ' '.join(new_tokens)
        return decoded
    
    def __call__(self, texts, **kwargs):
        token_idxs = [np.array(text.split(' ')).astype(int) for text in texts]
        if self.collapse:
            token_idxs = [self._collapse_numpy(tokens) for tokens in token_idxs]
        texts = [' '.join(tokens.astype(str)) for tokens in token_idxs]
        token_idxs = [list(tokens) for tokens in token_idxs]
        if self.spm is not None:
            token_idxs = [self.spm.Encode(''.join(chr(self.valid_unicode[t]) for t in tokens)) for tokens in token_idxs]
        token_idxs=[torch.tensor([self.blank] + tokens) for tokens in token_idxs]
        token_lengths = torch.tensor([len(idxs) for idxs in token_idxs])
        token_idxs = nn.utils.rnn.pad_sequence(token_idxs,batch_first=True, padding_value=self.pad_id)
        return token_idxs, token_lengths, texts
    
    def get_decoder(self):
        if self.spm is not None:
            return self._spm_decode #lambda tokens: ' '.join([str(t) for t in self._reverse(self.spm.DecodeIds(tokens))])
        else:
            return lambda tokens: ' '.join([str(t) for t in tokens])
    
    def get_remove_tokens(self):
        return [self.pad_id, self.blank]
    
    def pad_id(self):
        return self.pad_id
    
    def bos_id(self):
        return self.blank
    
    def eos_id(self):
        return self.blank

class HuBERTTokenizerPW(HuBERTTokenizer):
    def __init__(self, pre_tokenized=True, km_n=200, device='cuda',km_path=None, collapse=True, spm=None, **kwargs):
        super().__init__()
        if not pre_tokenized:
            assert km_n in [50, 100, 200], "Only km_n in [50, 100, 200] is supported"
            from torchpq.clustering import MinibatchKMeans
            self.km_model = MinibatchKMeans(km_n)
            self.km_model.load_state_dict(torch.load(km_path))
            self.km_model = self.km_model.eval().to(device)
            from transformers import Wav2Vec2Processor, HubertModel
            self.device = device
            self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
            self.speech_encoder = HubertModel.from_pretrained('facebook/hubert-large-ll60k').to(self.device)
            self.layer = 6  
        if spm is not None:
            import sentencepiece
            from pathlib import Path
            if not Path(spm).exists():
                spm = str(Path(__file__).parent.parent/'spm'/f'hubert-l6_km{km_n}_{spm}.model')
            
            spm =str(spm)
            self.spm = sentencepiece.SentencePieceProcessor(model_file=spm)
            self.valid_unicode = np.load(Path(__file__).parent.parent/'misc'/'valid_unicode.npy')
            self.reverse = {code:i for i,code in enumerate(self.valid_unicode)}
            spm_file_tag=Path(spm).stem
            if 'bpe' in spm_file_tag:
                vocab_size = int(spm_file_tag.split('bpe')[-1])
            elif 'unigram' in spm_file_tag:
                vocab_size = int(spm_file_tag.split('unigram')[-1])
            else:
                raise NotImplemented
            self.pad_id = vocab_size
            self.blank = vocab_size+1
            self.collapse=True
        else:
            self.spm = None 
            self.pad_id = km_n 
            self.blank = km_n+1
            self.collapse = collapse
            
        self.pre_tokenized = pre_tokenized
    
    def tokenize(self, wav, sr=16000):
        assert not self.pre_tokenized, "set pre_tokenized=False to load quantizer model"
        inputs=self.processor(wav,sampling_rate=sr, return_tensors="pt")
        inputs=inputs.to(self.device)
        with torch.no_grad():
            outputs = self.speech_encoder(**inputs,output_hidden_states=True)
        states=outputs.hidden_states[self.layer].squeeze(0).transpose(0,1)
        tokens = self.km_model.predict(states).cpu().numpy()
        return tokens
    
    
    

class SDHuBERTTokenizer(nn.Module):
    
    def __init__(self, pre_tokenized=True, km_n=4096, device='cuda', collapse=True, **kwargs):
        super().__init__()
        #import pdb
        #pdb.set_trace()
        if not pre_tokenized:
            import joblib
            from pathlib import Path
            from .sdhubert import SDHuBERT
            from transformers import Wav2Vec2Processor
            print('Loading model...')
            sdhubert_asset_path = Path(__file__).parent.parent/'sdhubert_asset'
            ckpt = str(sdhubert_asset_path/'sdhubert.ckpt')
            km_path = str(sdhubert_asset_path/'sd_hubert_l9-16384.pt')
            reducer_path = str(sdhubert_asset_path/f'sd_hubert_l9-16384to{km_n}.npy')
            self.speech_encoder = SDHuBERT(ckpt=ckpt).to(device)
            self.km_model = joblib.load(open(str(km_path), "rb"))
            self.km_model.verbose = False
            self.reducer = np.load(reducer_path)
            self.device = device
            self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
            self.layer=9
            
        self.pad_id = km_n 
        self.blank = km_n+1
        self.collapse = collapse
        self.pre_tokenized = pre_tokenized
        
    
    def get_feature(self, wav, sr=16000, output_all=False,):
        assert not self.pre_tokenized, "set pre_tokenized=False to load quantizer model"
        inputs=self.processor(wav,sampling_rate=sr, return_tensors="pt")
        inputs=inputs.to(self.device)
        with torch.no_grad():
            outputs = self.speech_encoder(inputs.input_values)
        if output_all:
            states=outputs['hidden_states']
        else:
            states=outputs['hidden_states'][self.layer].squeeze(0).cpu().numpy()
        
        return states
    
    def get_boundaries(self, wav, sr=16000, thr=3):
        assert not self.pre_tokenized, "set pre_tokenized=False to load quantizer model"
        inputs=self.processor(wav,sampling_rate=sr, return_tensors="pt")
        inputs=inputs.to(self.device)
        with torch.no_grad():
            outputs = self.speech_encoder(inputs.input_values)
        states=outputs['hidden_states'][11].squeeze(0).cpu().numpy()
        boundaries = (np.linalg.norm(states, axis=-1) <thr)*1.0
        return boundaries
    
    def tokenize(self, wav, sr=16000):
        assert not self.pre_tokenized, "set pre_tokenized=False to load quantizer model"
        inputs=self.processor(wav,sampling_rate=sr, return_tensors="pt")
        inputs=inputs.to(self.device)
        with torch.no_grad():
            outputs = self.speech_encoder(inputs.input_values)
        states=outputs['hidden_states'][self.layer].squeeze(0).cpu().numpy()
        tokens = self.reducer[self.km_model.predict(states)]
        return tokens
    
    @staticmethod
    def _collapse_numpy(tokens):
        is_continuous = tokens[1:]==tokens[:-1]
        return np.concatenate([tokens[:1], tokens[1:][~is_continuous]])
    
    def __call__(self, texts, **kwargs):
        token_idxs = [np.array(text.split(' ')).astype(int) for text in texts]
        if self.collapse:
            token_idxs = [self._collapse_numpy(tokens) for tokens in token_idxs]
        texts = [' '.join(tokens.astype(str)) for tokens in token_idxs]
        token_idxs=[torch.tensor([self.blank] + tokens) for tokens in token_idxs]
        token_lengths = torch.tensor([len(idxs) for idxs in token_idxs])
        token_idxs = nn.utils.rnn.pad_sequence(token_idxs,batch_first=True, padding_value=self.pad_id)
        return token_idxs, token_lengths, texts
    
    def get_decoder(self):
        return lambda tokens: ' '.join([str(t) for t in tokens])
    
    def get_remove_tokens(self):
        return [self.pad_id, self.blank]
    
    def pad_id(self):
        return self.pad_id
    
    def bos_id(self):
        return self.blank
    
    def eos_id(self):
        return self.blank
    
    
class SDHuBERTSegmenter(nn.Module):
    
    def __init__(self, syl_dur=0.2, ft_sr=50, merge_threshold=0.3, min_segment_len=0, 
                 min_cut_minimum=5, device='cuda', **kwargs):
        super().__init__()
        from pathlib import Path
        from .sdhubert import SDHuBERT
        from transformers import Wav2Vec2Processor
        from mincut import mincut
        import joblib
        sdhubert_asset_path = Path(__file__).parent.parent/'sdhubert_asset'
        ckpt = str(sdhubert_asset_path/'sdhubert.ckpt')
        self.speech_encoder = SDHuBERT(ckpt=ckpt).to(device).eval()
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        self.layer=9
        self.normcut_layer=11
        self.normcut_threshold = 2
        self.mincut = mincut
        self.syl_dur = syl_dur
        self.ft_sr = ft_sr
        self.wav_sr = 16000
        self.merge_threshold = merge_threshold
        self.min_segment_len = min_segment_len
        self.min_cut_minimum = min_cut_minimum
        km_path = sdhubert_asset_path/'sd_hubert_paper_16384.pt'
        reducer_path = sdhubert_asset_path/f'sd_hubert_paper_16384to4096.npy'
        if km_path.exists() and reducer_path.exists():
            self.km_model = joblib.load(open(str(km_path), "rb"))
            self.km_model.verbose = False
            self.reducer = np.load(reducer_path)
        else:
            print("No KMeans found")
            self.km_model = None
            self.reducer = None
            
    
    def mask_to_segment(self, mask):
        # mask: 1d-array of boolean mask
        valid_mask_ext = np.concatenate([np.zeros(1),mask*1.0,np.zeros(1)],0)

        turning = valid_mask_ext[1:]-valid_mask_ext[:-1]

        turn_on = np.nonzero(turning>0)[0]
        turn_off =np.nonzero((-turning)>0)[0]

        segments = np.array([[turn_on[i],turn_off[i]] for i in range(len(turn_on))])

        return segments

    def trim_segment(self, segments):
        trimmed_segments=[]
        for si,ei in segments:
            if ei-si>=self.min_segment_len:
                trimmed_segments.append([si,ei])
        trimmed_segments=np.array(trimmed_segments)
        return trimmed_segments


    def mincut_wrapper(self, feat):
        # feat: (T, d)
        num_syllable = int(np.ceil(len(feat)/self.ft_sr/self.syl_dur))
        
        ssm = feat@feat.transpose(1,0)
        ssm = ssm - np.min(ssm) + 1e-7 # make it non-negative
        seg_boundary_frame = self.mincut.min_cut(ssm, num_syllable+1) # +1 for the algo

        seg_boundary_frame_pairs_orig = [[l,r] for l, r in zip(seg_boundary_frame[:-1], seg_boundary_frame[1:])] # 
        seg_boundary_frame_pairs = [item for item in seg_boundary_frame_pairs_orig if item[1]-item[0] > 2]
        if len(seg_boundary_frame_pairs)==0: # this shouldn't happen though
            seg_boundary_frame_pairs = seg_boundary_frame_pairs_orig

        if len(seg_boundary_frame_pairs) >= 3:
            seg_boundary_frame_pairs = seg_boundary_frame_pairs_orig
            all_feat = [feat[round(l):round(r)].mean(0) for l,r in seg_boundary_frame_pairs]
            all_sim = [np.dot(l,r)/(np.linalg.norm(l)*np.linalg.norm(r)) for l,r in zip(all_feat[:-1], all_feat[1:])]
            min_id = np.argmax(all_sim)
            while all_sim[min_id] >= self.merge_threshold and len(seg_boundary_frame_pairs) >= 3:
                l_merge, r_merge = seg_boundary_frame_pairs[min_id], seg_boundary_frame_pairs[min_id+1]
                seg_boundary_frame_pairs = [pair for i, pair in enumerate(seg_boundary_frame_pairs) if i != min_id and i != min_id+1]
                seg_boundary_frame_pairs.insert(min_id, [l_merge[0], r_merge[1]])
                all_feat = [feat[round(l):round(r)].mean(0) for l,r in seg_boundary_frame_pairs]
                all_sim = [np.dot(l,r)/(np.linalg.norm(l)*np.linalg.norm(r)) for l,r in zip(all_feat[:-1], all_feat[1:])]
                min_id = np.argmax(all_sim)

        feat = [feat[round(l):round(r)].mean(0) for l,r in seg_boundary_frame_pairs]

        return feat, seg_boundary_frame_pairs


    def extract(self, wav):
        if isinstance(wav, str):
            wav, sr = sf.read(wav)
            if sr != self.wav_sr:
                wav = librosa.resample(wav,orig_sr=sr,target_sr=self.wav_sr)
        inputs=self.processor(wav,sampling_rate=self.wav_sr, return_tensors="pt",padding=True,return_attention_mask=True)
        inputs=inputs.input_values.to(self.device)
        with torch.no_grad():
            outputs = self.speech_encoder(inputs,np.array([len(wav)/16000]))
        states=outputs #['hidden_states']
        
        return states
    
    def get_feature_mask(self, wav):
        
        inputs=self.processor(wav,sampling_rate=self.wav_sr, return_tensors="pt",padding=True,return_attention_mask=True)
        inputs=inputs.input_values.to(self.device)
        with torch.no_grad():
            outputs = self.speech_encoder(inputs,np.array([len(wav)/16000]))
        states=outputs['hidden_states'][self.layer].squeeze(0).cpu().numpy()
        norm = outputs['hidden_states'][self.normcut_layer].squeeze(0).cpu().numpy()
        valid_mask=np.linalg.norm(norm,axis=1)>self.normcut_threshold
        states[~valid_mask] = 0
        return states, valid_mask
    

    def segment(self, wav, keep_ft_sr=True):
        if isinstance(wav, str):
            wav, sr = sf.read(wav)
            if sr != self.wav_sr:
                wav = librosa.resample(wav,orig_sr=sr,target_sr=self.wav_sr)
        features, mask = self.get_feature_mask(wav)
        segments = self.trim_segment(self.mask_to_segment(mask))
        
        boundaries=[]
        pooled_feat=[]
        for segment in segments:
            if (segment[1]-segment[0])<self.min_cut_minimum:
                boundaries_=[(segment-segment[0])]
                pooled_feat_=[features[segment[0]:segment[1]].mean(0)]
            else:
                pooled_feat_, boundaries_ = self.mincut_wrapper(features[segment[0]:segment[1]])
            for bi,(bd,ft_) in enumerate(zip(boundaries_,pooled_feat_)):
                if np.isnan(np.sum(ft_)):
                    continue
                boundaries.append(bd+segment[0])
                pooled_feat.append(ft_)
                
        boundaries=np.stack(boundaries)
        if not keep_ft_sr:
            boundaries=boundaries*1.0/self.ft_sr
        pooled_feat=np.stack(pooled_feat)
        
        return features, pooled_feat, boundaries
    
    def tokenize(self, wav, skip_quantizer=False, skip_reducer=False):
        features, pooled_feat, boundaries = self.segment(wav)
        if skip_quantizer:
            return pooled_feat
        tokens = self.km_model.predict(pooled_feat)
        if not skip_reducer:
            tokens =self.reducer[tokens]
        return tokens
    
class TM1KTokenizer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        from pathlib import Path
        lexicons = Path(__file__).parent.parent/'misc'/'tm1k_lexicon.txt'
        with open(lexicons ,'r') as f:
            lexicons =f.readlines()
        self.lexicons = [l.rstrip() for l in lexicons]
        self.lex2idx = {l:i for i,l in enumerate(self.lexicons )}
        self.pad_id = len(self.lexicons) 
        self.blank =  len(self.lexicons)+1
        self.vocab_size = len(self.lexicons)+2
        
        
    def __call__(self, texts, **kwargs):
        texts=[text.lower() for text in texts]
        token_idxs = [[self.lex2idx[t] for t in text.split(' ') if t != ''] for text in texts]
        token_idxs = [torch.tensor([self.blank] + idxs) for idxs in token_idxs]
        token_lengths = torch.tensor([len(idxs) for idxs in token_idxs])
        
        token_idxs = nn.utils.rnn.pad_sequence(token_idxs,batch_first=True, padding_value=self.pad_id)
        return token_idxs, token_lengths, texts
    
    def get_decoder(self):
        return lambda tokens: ''.join([self.lexicons[i] for i in tokens])
    
    def get_remove_tokens(self):
        return [self.pad_id, self.blank]
    
    def pad_id(self):
        return self.pad_id
    
    def bos_id(self):
        return self.blank
    
    def eos_id(self):
        return self.blank
    