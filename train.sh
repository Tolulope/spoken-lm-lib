#!/usr/bin/env bash
export HF_HOME= 
export TOKENIZERS_PARALLELISM=true

python3 train.py --config=configs/config.yml \
                 wandb.id=asr50h-expanse-10-epochs-again-warm7\
                 model.is_qformer=True \
                 model.lm_name_or_path=CohereForAI/aya-23-8B \
                 model.speech_model_name_or_path=facebook/hubert-large-ls960-ft \
                 prompts.train_prompts_path="prompts/train_prompt.json" \
                 dataset.is_json=True \
                 dataset.train_dataset_path="/nlp/scr/tolulope/mm-lib/multi_asr_train.json" \
                 dataset.dev_dataset_path="/nlp/scr/tolulope/mm-lib/multi_asr_dev.json" \
                 dataset.test_dataset_path="/nlp/scr/tolulope/mm-lib/multi_asr_test.json" \
                 dataset.shuffle_train_dataset=True \
                 dataset.shuffle_dev_dataset=False \
                 dataset.shuffle_test_dataset=False \
                 lightningargs.output_dir="/juice2/scr2/nlp/speech-data/multimodal-models/salmonn/stage1-asr-50h-10-test/" \
                 lightningargs.per_device_train_batch_size=8 \
                 lightningargs.min_epochs=1 \
                 lightningargs.max_epochs=5 \
                 lightningargs.val_check_interval=0.5 \
                 lightningargs.check_val_every_n_epoch=1 \
                 lightningargs.reload_dataloaders_every_n_epochs=0 \
                 lightningargs.limit_train_batches=1.0 \
                 lightningargs.optims.warmup_steps=375 \
                 lightningargs.accumulate_grad_batches=4 \

