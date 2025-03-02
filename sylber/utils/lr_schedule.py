import math

class LRLAMBDA(object):
    def __init__(self, warmup_steps, total_steps, min_factor=0.05, hold_steps=0 ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_factor = min_factor
        self.hold_steps = hold_steps
        
    
    def __call__(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps)) * 1
        else:
            net_step = max(0,step-self.warmup_steps-self.hold_steps)

            return max(
                self.min_factor, 1.0 - float(net_step) / float(max(1, net_step))
            ) * 1
            

class COSLRLAMBDA(object):
    def __init__(self, warmup_steps, total_steps, min_factor=0.05, hold_steps=0 ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_factor = min_factor
        self.hold_steps = hold_steps
        
    
    def __call__(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps)) * 1
        elif step> (self.total_steps + self.hold_steps+self.warmup_steps):
            return self.min_factor
        else:
            net_step = max(0,step-self.warmup_steps-self.hold_steps)
            return self.min_factor+(1-self.min_factor)*(1+math.cos( float(net_step)/float(max(1, self.total_steps)) *math.pi))/2
