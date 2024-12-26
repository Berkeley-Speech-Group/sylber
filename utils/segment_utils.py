import numpy as np
import torch
import torch.nn as nn


class Thresholder(nn.Module):
    def __init__(self, signal_mean=None, signal_var=None, noise_mean=None, noise_var=None, decay=0.9999, eta=1,
                threshold=None):
        super().__init__()
        if threshold is None:
            self.signal_mean = nn.parameter.Parameter(torch.ones(1)*signal_mean,requires_grad=False)
            self.signal_var = nn.parameter.Parameter(torch.ones(1)*signal_var,requires_grad=False)
            self.noise_mean = nn.parameter.Parameter(torch.ones(1)*noise_mean,requires_grad=False)
            self.noise_var = nn.parameter.Parameter(torch.ones(1)*noise_var,requires_grad=False)
            self.decay = decay
            self.eta = eta
            self.threshold =  None
        else:
            self.signal_mean = nn.parameter.Parameter(torch.ones(1),requires_grad=False)
            self.signal_var = nn.parameter.Parameter(torch.ones(1),requires_grad=False)
            self.noise_mean = nn.parameter.Parameter(torch.ones(1),requires_grad=False)
            self.noise_var = nn.parameter.Parameter(torch.ones(1),requires_grad=False)
            self.decay = decay
            self.eta = eta
            self.threshold =  nn.parameter.Parameter(torch.ones(1)*threshold,requires_grad=False)
        
    def get_threshold(self):
        if self.threshold is not None:
            return self.threshold
        else:
            with torch.no_grad():
                
                mu_S = self.signal_mean  # Mean of noise
                sigma_S = (self.signal_var+1e-8)**.5  # Standard deviation of noise
                mu_N = self.noise_mean  # Mean of signal
                sigma_N = (self.noise_var+1e-8)**.5  # Standard deviation of signal
                eta = self.eta
                A = sigma_S**2 - sigma_N**2
                B = -2*sigma_S**2*mu_N + 2*sigma_N**2*mu_S
                C = sigma_S**2*mu_N**2 - sigma_N**2*mu_S**2 - 2*sigma_N**2*sigma_S**2*(np.log(eta) + torch.log(sigma_S/sigma_N))
    
                if A != 0:
                    discriminant = B**2 - 4*A*C
                    if discriminant > 0:
                        threshold = (-B + ((mu_S>mu_N)*1.0)*torch.sqrt(discriminant)) / (2*A)
                    elif discriminant == 0:
                        threshold = -B / (2*A)
                else:
                    # When A is zero, solve the linear equation Bx + C = 0
                    if B != 0:
                        threshold = -C / B
            return threshold

    
    def update_stats(self, signal, noise):
        if self.threshold is not None:
            return
        with torch.no_grad():
            if signal is not None:
                self.signal_mean.data = self.decay*self.signal_mean.data+(1-self.decay)*signal.mean()
                self.signal_var.data = self.decay*self.signal_var.data+(1-self.decay)*((signal-self.signal_mean.data)**2).mean()
            if noise is not None:
                self.noise_mean.data = self.decay*self.noise_mean.data+(1-self.decay)*noise.mean()
                self.noise_var.data = self.decay*self.noise_var.data+(1-self.decay)*((noise-self.noise_mean.data)**2).mean()



def cossim(x,y):
    return (x*y).sum(-1)/(((x**2).sum(-1)+1e-8)**.5)/(((y**2).sum(-1)+1e-8)**.5)


def get_segment(states, normthreshold, mergethreshold,norms=None):
    # states: (L, d)
    if norms is None:
        norms = ((states**2).sum(-1)+1e-8)**.5
    mask = norms>=normthreshold

    curr = 0
    seg_cnt = 0
    segments = []
    s=-1
    midboundaries = []
    for i in range(len(states)):
        if ~mask[i]:
            seg_cnt = 0
            if s>-1:
                segments.append([s,i])
            s = -1
            curr = 0
            seg_cnt = 0
        else:
            if seg_cnt ==0:
                curr = states[i]
                seg_cnt+=1
                s = i
            else:
                sim = cossim(curr, states[i])
                if sim >= mergethreshold:
                    curr = (curr*seg_cnt+states[i])/(seg_cnt+1)
                    seg_cnt += 1
                else:
                    curr = states[i]
                    seg_cnt += 1
                    segments.append([s,i])
                    midboundaries.append([i,len(segments)-1])
                    s = i
    if s>-1:
        segments.append([s,len(states)])
        
    merged = []
    for bd, segi in midboundaries:
        if segi >= len(segments)-1:
            continue
        if cossim(states[segments[segi][0]:segments[segi][1]].mean(0),
                  states[segments[segi+1][0]:segments[segi+1][1]].mean(0))>=mergethreshold:
            segments[segi+1]=[segments[segi][0],segments[segi+1][1]]
            merged.append(segi)
            continue
        s = max(segments[segi][0],bd-max(1,(segments[segi][1]-segments[segi][0])//2))
        bd = min(segments[segi+1][1],bd+max(1,(segments[segi+1][1]-segments[segi+1][0])//2))
        prev_center = states[segments[segi][0]:segments[segi][1]].mean(0)
        next_center = states[segments[segi+1][0]:segments[segi+1][1]].mean(0)
        sim_prev = cossim(states[s:bd],prev_center[None,:])
        sim_next = cossim(states[s:bd],next_center[None,:])
        sim_sweep = [(sim_prev[:i].sum()+sim_next[i:].sum()) for i in range(0,bd-s)]
        opt_b = np.arange(s,bd)[np.argmax(sim_sweep)]
        segments[segi] = [segments[segi][0],opt_b]
        segments[segi+1] = [opt_b,segments[segi+1][1]]  
    
    segments = [segment for segi, segment in enumerate(segments) if segi not in merged]                
    return np.array(segments)
