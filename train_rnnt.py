from sylber.dataset.librispeech_asr import SpeechTextDataModule
from asr.ssl_rnnt import ASRTrainer
import lightning as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import hydra
from pathlib import Path
import torch

torch.set_float32_matmul_precision('medium')

@hydra.main(config_path='asr_configs') #, config_name='sylber_base')
def main(cfg):
    
    # datamodule
    datamodule = SpeechTextDataModule(**cfg['data'])

    # model
    model = ASRTrainer(**cfg['model'])
    if 'speech_model_ckpt' in cfg.keys() and cfg['speech_model_ckpt'] != None:
        state_dict = torch.load(cfg['speech_model_ckpt'], map_location='cpu')
        model.net.speech_model.load_state_dict(state_dict, strict=False )
        print("Pre-trained checkpoint loaded")

    if 'model_ckpt' in cfg.keys() and cfg['model_ckpt'] != None:
        state_dict = torch.load(cfg['model_ckpt'], map_location='cpu')['state_dict']
        model.load_state_dict(state_dict, strict=False)
        print("A whole pre-trained checkpoint loaded")

    if 'load_ema' in cfg.keys() and cfg['load_ema'] != None:
        state_dict = torch.load(cfg['load_ema'], map_location='cpu')['ema']
        model.net.speech_model.load_state_dict(state_dict)
        print("EMA checkpoint loaded")
    
    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # checkpoint best
    checkpoint_callback_topk = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename='best-{epoch}-{val_loss:.2f}'
    )

    # checkpoint best
    checkpoint_callback_top_wer = ModelCheckpoint(
        monitor="val_wer",
        save_top_k=1,
        mode="min",
        filename='best-{epoch}-{val_wer:.3f}'
    )

    
    # checkpoint every N epochs
    checkpoint_callback_by_epoch = ModelCheckpoint(
        every_n_epochs=cfg['checkpoint_epoch'],
    )
    
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
    
    callbacks  = [checkpoint_callback_topk, checkpoint_callback_by_epoch,
                  checkpoint_callback_top_wer,
                  LearningRateMonitor(logging_interval='step')]
    
    if 'earlystop_metric' in cfg.keys() and cfg['earlystop_metric'] is not None:
        early_stop_callback = EarlyStopping(monitor=cfg['earlystop_metric'], min_delta=0.0001, patience=5, verbose=False, mode="min")
        callbacks.append(early_stop_callback)
        
    scaler = torch.cuda.amp.GradScaler()
    
    trainer = pl.Trainer(devices=gpus,
                         accelerator="gpu",
                         strategy="ddp_find_unused_parameters_true",
                         max_steps = cfg['max_steps'],
                         num_sanity_val_steps=0,
                         check_val_every_n_epoch=cfg['check_val_every_n_epoch'],
                         val_check_interval=cfg['val_check_interval'],
                         limit_val_batches=cfg['limit_val_batches'],
                         callbacks=callbacks,
                         gradient_clip_val=0.5,
                         default_root_dir=cfg.get('name', 'noname'),
                        )

    # fit model
    trainer.fit(model,datamodule,ckpt_path=cfg['resume_ckpt'],)

if __name__ =='__main__':
    main()
