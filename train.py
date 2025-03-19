import os
import omegaconf as oc
import torch
import transformers
import wandb
from torch.utils.data import DataLoader
# from transformers import (
#     WhisperFeatureExtractor,
# )

import lightning.pytorch as pl
from lightning.pytorch.plugins.environments import LightningEnvironment
from lightning.pytorch.plugins.precision import MixedPrecisionPlugin
from lightning.pytorch.callbacks import ModelCheckpoint,  LearningRateMonitor

from adapters import SavingCallback
from dataset import SpeechLMDataset, SpeechLMCollator
from speechlm import SpeechLM
from task_batch import CustomBatchSampler
import utils


config = oc.OmegaConf.from_cli()

assert '--config' in config.keys(), """\n
    Please supply a base config file, e.g. 'python train.py --config=CONFIG_FILE.yml'.

    You can then over-ride config parameters, e.g. 'python train.py --config=CONFIG_FILE.yml lightningargs.min_epochs=1'
"""
utils.announce("Configuring model")

config, wandb_logger = utils.make_config(config)


if config['model']['slm_proj_ckpt']:
    model = SpeechLM.load_from_checkpoint(config['model']['slm_proj_ckpt'], config=config, map_location=torch.device("cpu"), strict=False)
else:
    model = SpeechLM(config)


wandb_logger.watch(model, log_freq=1)

# speech_feat_extractor = WhisperFeatureExtractor.from_pretrained(config['model']['speech_model_name_or_path'])

data_collator = SpeechLMCollator(tokenizer=model.speech_feat_extractor)

if config['model']['task_batching']:
    sampler = CustomBatchSampler(
        SpeechLMDataset(config, config['dataset']['train_dataset_path'], limit_dev_dataset=0),
        batch_size=config['lightningargs']['per_device_train_batch_size'],
    )
elif config['model']['dynamic_batching']:
    from speechbrain.dataio.sampler import DynamicBatchSampler, DistributedSamplerWrapper
    # from dynamic_batch import DynamicBatchSampler, DynamicBatchSamplerDDP
    train_ds_samp = SpeechLMDataset(config, config['dataset']['train_dataset_path'], limit_dev_dataset=0)
    if config['lightningargs']['devices'] > 1:
        # sampler = DistributedSamplerWrapper(sampler)
        # sampler = DynamicBatchSampler(
        #         dataset=train_ds_samp,
        #         lengths_list=[i*16_000 for i in train_ds_samp.ds['len'] ] ,
        #         num_buckets=200,
        #         max_batch_length=16_000 * 30 * 8,
        #         max_batch_ex=7,
        #         shuffle=True,
        #         batch_ordering='random')
        sampler=None
    else:
        sampler = DynamicBatchSampler(
                dataset=train_ds_samp,
                lengths_list=[i*16_000 for i in train_ds_samp.ds['len'] ] ,
                num_buckets=200,
                max_batch_length=16_000 * 30 * 8,
                max_batch_ex=7,
                shuffle=True,
                batch_ordering='random')

    # if config['lightningargs']['devices'] > 1:
    #     sampler = DistributedSamplerWrapper(sampler)
    # sampler = CustomBatchSampler(
    #     SpeechLMDataset(config, config['dataset']['train_dataset_path'], limit_dev_dataset=0),
    #     batch_size=config['lightningargs']['per_device_train_batch_size'],
    # )
    # print("Finished loading custom batch sampler")
else:
    sampler=None

if config['model']['task_batching']:
    train_ds = DataLoader(dataset=SpeechLMDataset(config, config['dataset']['train_dataset_path'], limit_dev_dataset=0), 
                        collate_fn=data_collator, 
                        num_workers=config['lightningargs']['dataloader_num_workers'], 
                        shuffle=config['dataset']['shuffle_train_dataset'],
                        batch_sampler=sampler)
elif config['model']['dynamic_batching']:
    train_ds = DataLoader(dataset=SpeechLMDataset(config, config['dataset']['train_dataset_path'], limit_dev_dataset=0), 
                        collate_fn=data_collator, 
                        num_workers=config['lightningargs']['dataloader_num_workers'], 
                        batch_sampler=sampler)
    first_batch = next(iter(train_ds))
    print(first_batch)

else:
    train_ds = DataLoader(dataset=SpeechLMDataset(config, config['dataset']['train_dataset_path'], limit_dev_dataset=0), 
                        collate_fn=data_collator, 
                        batch_size=config['lightningargs']['per_device_train_batch_size'], 
                        num_workers=config['lightningargs']['dataloader_num_workers'], 
                        shuffle=config['dataset']['shuffle_train_dataset'],
                        batch_sampler=sampler)

val_ds = DataLoader(dataset=SpeechLMDataset(config, config['dataset']['dev_dataset_path'], 
                    limit_dev_dataset=config['dataset']['limit_dev_dataset']), 
                    collate_fn=data_collator, 
                    batch_size=config['lightningargs']['per_device_eval_batch_size'], 
                    num_workers=config['lightningargs']['dataloader_num_workers'], 
                    shuffle=config['dataset']['shuffle_dev_dataset'])


checkpoint_callback = ModelCheckpoint(dirpath=config['lightningargs']['output_dir'], save_top_k=3, verbose=True, monitor="val_loss")
lr_monitor = LearningRateMonitor(logging_interval='step')


trainer = pl.Trainer(
        min_epochs=config['lightningargs']['min_epochs'],
        max_epochs=config['lightningargs']['max_epochs'],
        val_check_interval=config['lightningargs']['val_check_interval'],
        check_val_every_n_epoch=config['lightningargs']['check_val_every_n_epoch'],
        reload_dataloaders_every_n_epochs=config['lightningargs']['reload_dataloaders_every_n_epochs'],
        limit_train_batches=config['lightningargs']['limit_train_batches'],
        accumulate_grad_batches=config['lightningargs']['accumulate_grad_batches'],
        default_root_dir=config['lightningargs']['output_dir'],
        accelerator=config['lightningargs']['accelerator'],
        devices=config['lightningargs']['devices'],
        strategy=config['lightningargs']['strategy'],
        use_distributed_sampler=config['lightningargs']['use_distributed_sampler'],
        logger=wandb_logger,
        precision="16-mixed",
        plugins=[LightningEnvironment()],
        callbacks=[SavingCallback(), checkpoint_callback, lr_monitor],
)

utils.announce("Beginning training")

torch.cuda.empty_cache()

trainer.fit(model, train_dataloaders=train_ds, val_dataloaders=val_ds)
