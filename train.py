from sylber.dataset.collective_audio_segment import SpeechDataModule
from sylber.model.sylber import SylberTrainer
import lightning as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import hydra
import torch
from collections import OrderedDict
from weakref import proxy


class ModelCheckpointWithEMA(ModelCheckpoint):

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        trainer.save_checkpoint(filepath, self.save_weights_only)
        model = trainer.lightning_module.net
        ema_dict= OrderedDict()
        ema_dict['ema']=model.ema.model.state_dict()
        if model.lm_ema is not None:
            ema_dict['lm_ema']=model.lm_ema.model.state_dict()
        if model.input_ema is not None:
            ema_dict['input_ema']=model.input_ema.model.state_dict()
        if model.logit_ema is not None:
            ema_dict['logit_ema']=model.logit_ema.model.state_dict()
        torch.save(ema_dict,"ema_dict.ckpt")

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))

@hydra.main(config_path='sylber_configs', config_name='sylber_base')     
def main(cfg):
    
    print(cfg)
    
    # datamodule
    datamodule = SpeechDataModule(**cfg['data'])

    # model
    model = SylberTrainer(**cfg['model'])
    if 'speech_model_ckpt' in cfg.keys() and cfg['speech_model_ckpt'] != None:
        state_dict = torch.load(cfg['speech_model_ckpt'], map_location='cpu')
        model.net.speech_model.load_state_dict(state_dict, strict=False )
        print("Pre-trained checkpoint loaded")

    if 'model_ckpt' in cfg.keys() and cfg['model_ckpt'] != None:
        state_dict = torch.load(cfg['model_ckpt'], map_location='cpu')['state_dict']
        try:
            model.load_state_dict(state_dict, strict=False)
        except:
            print("Can't load LM. Removing the weights.")
            new_dict= OrderedDict()
            for name, state in state_dict.items():
                if 'net.language_model' not in name and 'net.logit' not in name and 'net.input_linear' not in name :
                    new_dict[name] = state
            model.load_state_dict(new_dict, strict=False)
        print("Previous stage checkpoint loaded")

    
        
    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # checkpoint every N epochs
    checkpoint_callback_by_epoch = ModelCheckpointWithEMA(
        every_n_epochs=cfg['checkpoint_epoch'],
    )
    checkpoint_callback_last5 = ModelCheckpoint(save_top_k=5, mode='max', monitor='epoch')
    
    # Trainer
    if cfg['gpus'] is not None:
        if not isinstance(cfg['gpus'],list):
            try:
                gpus = [int(cfg['gpus'])]
            except:
                gpus = [int(x) for x in cfg['gpus'].split(',')]
        else:
            gpus = cfg['gpus']
    else:
        gpus= None
    
    callbacks  = [checkpoint_callback_last5,
                  checkpoint_callback_by_epoch,
                  LearningRateMonitor(logging_interval='step')]
    
    scaler = torch.cuda.amp.GradScaler()
    
    trainer = pl.Trainer(devices=gpus,
                         accelerator="gpu",
                         strategy="ddp_find_unused_parameters_true",
                         max_steps = cfg['max_steps'],
                         num_sanity_val_steps=0,
                         check_val_every_n_epoch=cfg['check_val_every_n_epoch'],
                         limit_val_batches=cfg['limit_val_batches'],
                         callbacks=callbacks,
                         gradient_clip_val=0.5,
                         default_root_dir=cfg.get('name', 'noname'),
                         accumulate_grad_batches=cfg['accumulate_grad_batches'],
                        )

    # fit model
    trainer.fit(model,datamodule,ckpt_path=cfg['resume_ckpt'],)

if __name__ =='__main__':
    main()
