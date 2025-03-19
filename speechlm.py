"""A one-line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.function_bar()
"""


import logging
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model

from transformers import (
    AutoModel,
    AutoModelForCausalLM, 
    WhisperModel, 
    AutoTokenizer, 
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoModelForSpeechSeq2Seq,
    StoppingCriteriaList
)
from transformers.models.auto.modeling_auto import MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_FOR_CTC_MAPPING_NAMES 
import lightning.pytorch as pl
from torch import optim
from Qformer import BertConfig, BertLMHeadModel
from beats.BEATs import BEATsConfig, BEATs
from utils import StoppingCriteriaSub
from lightning.pytorch.loggers import WandbLogger
import jiwer
from sacrebleu.metrics import BLEU
import numpy as np
import wandb
from torchtune.training import get_cosine_schedule_with_warmup 


class SpeechLM(pl.LightningModule):

    # Taken from SALMONN code (refer to SALMONN code)
    def init_speech_Qformer(cls, num_query_token, speech_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = speech_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def device(self):
        return list(self.parameters())[0].device

    def setup_speech(self, config):

        speech_model = AutoModel.from_pretrained(config['model']['speech_model_name_or_path'], device_map = 'cpu')
        speech_feat_extractor = AutoFeatureExtractor.from_pretrained(config['model']['speech_model_name_or_path'], device_map = 'cpu')

        if speech_model.config.model_type in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES.keys():
                speech_encoder = speech_model.encoder
                speech_d_model = speech_model.config.d_model

        if speech_model.config.model_type in MODEL_FOR_CTC_MAPPING_NAMES.keys():
            print(speech_model)
            speech_encoder = speech_model
            speech_d_model = speech_model.config.intermediate_size

        ln_speech = nn.LayerNorm(speech_d_model)
        if config['model']['freeze_speech_model']:
            for name, param in speech_encoder.named_parameters():
                param.requires_grad = False
            speech_encoder.eval()

        return speech_feat_extractor, speech_encoder, ln_speech, speech_d_model

    def setup_lm(self, config, lm_tokenizer):
        language_model = AutoModelForCausalLM.from_pretrained(config['model']['lm_name_or_path'],torch_dtype=torch.float16, device_map = 'cpu')
        language_model.resize_token_embeddings(len(lm_tokenizer))
        if config['model']['freeze_lm']:
            for name, param in language_model.named_parameters():
                param.requires_grad = False
        
        return language_model



    def setup_lm_tokenizer(self, config):
        lm_tokenizer = AutoTokenizer.from_pretrained(config['model']['lm_name_or_path'])
        lm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        lm_tokenizer.padding_side = "right"

        return lm_tokenizer


    def setup_audio(self, config):
            beats_ckpt = torch.load(config['model']['audio_model_name_or_path'], map_location='cpu')
            beats_cfg = BEATsConfig(beats_ckpt['cfg'])
            audio_model = BEATs(beats_cfg)
            ln_audio = nn.LayerNorm(audio_model.cfg.encoder_embed_dim)
            if config['model']['freeze_audio_model']:
                for name, param in audio_model.named_parameters():
                    param.requires_grad = False
                audio_model.eval()
                # logging.info("freeze BEATs")
            return audio_model, ln_audio
        # self.adapter =


    def __init__(self, config):

        super().__init__()

        self.window_level_Qformer = config['model']['is_window_level_qformer']
        self.second_per_window = config['model']['second_per_window']
        self.second_stride = config['model']['second_stride']
        self.lora = config['model']['peft_config']['lora']
        self.multi_prompt = config['prompts']['multi_prompt']
        self.use_speech_Qformer = config['model']['is_qformer']
        self.max_txt_len = config['prompts']['max_txt_len']
        if config['dataset']['test_set_tasks']['lang_id_options']:
            self.lang_id_options = config['dataset']['test_set_tasks']['lang_id_options']
        self.config = config

        
        print('Loading Speech Model')
        self.speech_feat_extractor, self.speech_encoder, self.ln_speech, self.speech_d_model = self.setup_speech(config)

        if config['model']['audio_model_name_or_path']:
            print('Loading Audio Model')
            self.audio_model, self.ln_audio = self.setup_audio(config)


        print('Loading Language Model Tokenizer')
        self.lm_tokenizer = self.setup_lm_tokenizer(config)
      

        print('Loading Language Model')
        self.language_model = self.setup_lm(config, self.lm_tokenizer)


        if config['model']['is_qformer']:
            print('Loading Q Former')
            if self.audio_model:
                self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
                    num_query_token=config['model']['num_speech_query_token'], speech_width=self.speech_d_model + self.audio_model.cfg.encoder_embed_dim
                )
            else:
                self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
                    num_query_token=config['model']['num_speech_query_token'], speech_width=self.speech_d_model
                )
            self.speech_Qformer.bert.embeddings.word_embeddings = None
            self.speech_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.speech_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.speech_Qformer.cls = None
            if config['model']['freeze_speech_QFormer']:
                for name, param in self.speech_Qformer.named_parameters():
                    param.requires_grad = False
                self.speech_Qformer.eval()
                self.speech_query_tokens.requires_grad = False
                # logging.info("freeze Speech QFormer")

            # logging.info('Loading speech LLAMA proj')
            self.speech_llama_proj = nn.Linear(
                self.speech_Qformer.config.hidden_size, self.language_model.config.hidden_size
            )
                # self.load_state_dict(speech_llama_proj_weight['model'], strict=False)
            if config['model']['freeze_slm_proj_ckpt']:
                for name, param in self.speech_llama_proj.named_parameters():
                    param.requires_grad = False
                self.speech_llama_proj.eval()
                # logging.info("freeze speech LLAMA proj")
        
        if config['model']['is_transformer']:
            print('Loading Transformer')
            # audio_model.cfg.encoder_embed_dim
            if self.audio_model:
                self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.speech_d_model + self.audio_model.cfg.encoder_embed_dim, nhead=config['model']['num_transformer_heads'], batch_first=True)
            else:
                self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.speech_d_model, nhead=config['model']['num_transformer_heads'], batch_first=True)
            self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=config['model']['num_transformer_layers'])
            if self.audio_model:
                self.speech_llama_proj = nn.Linear(
                    self.speech_encoder.config.d_model + self.audio_model.cfg.encoder_embed_dim, self.language_model.config.hidden_size
                )
            else:
                self.speech_llama_proj = nn.Linear(
                    self.speech_encoder.config.d_model, self.language_model.config.hidden_size
                )
                
        
        print('Loading PEFT Model')
        if config['model']['peft']:
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=config['model']['peft_config']['inference_mode'], 
                r=config['model']['peft_config']['r'], 
                lora_alpha=config['model']['peft_config']['lora_alpha'], 
                lora_dropout=config['model']['peft_config']['lora_dropout'],
                target_modules = config['model']['peft_config']['target_modules']
                
            )
            self.language_model = get_peft_model(self.language_model, self.peft_config)
            self.language_model.print_trainable_parameters()

        
        if config['model']['slm_proj_ckpt']:
            logging.info("Loading speech LLAMA proj from {}".format(config['model']['slm_proj_ckpt']))
            checkpoint_state_dict = torch.load(config['model']['slm_proj_ckpt'])['state_dict']
            # print(checkpoint['state_dict'].keys())
            # speech_llama_proj_weight = torch.load(config['model']['slm_proj_ckpt'], map_location="cpu")
            # print(speech_llama_proj_weight)
            model_state_dict = self.state_dict()
            for key in checkpoint_state_dict.keys():
                model_state_dict[key] = checkpoint_state_dict[key]
                self.load_state_dict(model_state_dict)


        
        # prepare prompts
        self.prompt_dict = {}
        if config['prompts']['train_prompts_path']:
            try:
                raw_prompts = json.load(open(config['prompts']['train_prompts_path'], "r"))
            except:
                print("Failed to load prompt! Try to use utf-8 encoding.")
                raw_prompts = json.load(open(config['prompts']['train_prompts_path'], "r", encoding='utf-8'))
            for task in raw_prompts.keys():
                # filted_prompts = [raw_prompt for raw_prompt in raw_prompts[task] if "<SpeechHere>" in raw_prompt]
                filted_prompts = [raw_prompt for raw_prompt in raw_prompts[task]]
                self.prompt_dict[task] = [config['prompts']['prompt_template'].format(p) for p in filted_prompts]
            print("Loading training prompts done!")
            self.end_sym = config['prompts']['end_sym']
            self.prompt_template = config['prompts']['prompt_template']

        # if config['model']['speechlm_checkpoint']:
        #     ckpt = torch.load(config['model']['speechlm_checkpoint'], map_location="cpu")
        #     # print(ckpt.keys())
        #     # self.model.load_state_dict(ckpt['model'], strict=False)
        #     self.language_model.load_state_dict(ckpt['model'], strict=False)
        #     self.speech_Qformer.load_state_dict(ckpt['model'], strict=False)
        #     self.llama_proj.load_state_dict(ckpt['model'], strict=False)

        
        self.asr_columns = ["Truth", "Transcription", "WER"]
        self.asr_table = wandb.Table(columns=self.asr_columns)
        # print(self.asr_table)
        # key="ASR", 
        self.ast_columns = ["Truth", "Translation", "BLEU"]
        self.ast_table = wandb.Table(columns=self.ast_columns)
        # key="AST", 
        self.langid_columns = ["Truth", "Lang Prediction", "Accuracy"]
        self.lid_table = wandb.Table(columns=self.langid_columns)
        # key="Lang ID", 
        self.wer_refs = []
        self.wer_hyps = []
        self.wer_ind = []

        self.bleu_syss = []
        self.bleu_refs = []
        self.bleu_ind = []

        self.lang_id_acc = []
        self.lang_truth = []
        self.lang_selected = []






    def encode_speech_and_audio(self,spec, audio_array, audio_array_paddded, audio_padding_mask=None):
        if self.config['model']['speech_model_return_last']:
            speech_embeds = self.speech_encoder(spec, return_dict=True).last_hidden_state
        else:
            speech_embeds = self.speech_encoder(spec, return_dict=True, output_hidden_states=True).hidden_states[self.config['model']['speech_model_return_last'] + 1]

        if self.audio_model:
            audio_embeds, _ = self.audio_model.extract_features(audio_array, padding_mask=audio_padding_mask, feature_only=True)
        else:
            audio_embeds = None

        return speech_embeds, audio_embeds


    def adapt_qformer(self, speech_embeds, audio_embeds=None):
        # taken from SALMONN
        # with self.maybe_autocast():
        if self.use_speech_Qformer:
            speech_embeds = self.ln_speech(speech_embeds)
            if audio_embeds is not None:
                audio_embeds = self.ln_audio(audio_embeds)
                if audio_embeds.size(1) < speech_embeds.size(1):
                    audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
                elif audio_embeds.size(1) > speech_embeds.size(1):
                    speech_embeds = F.pad(speech_embeds, (0, 0, 0, audio_embeds.size(1) - speech_embeds.size(1)))
                speech_embeds = torch.cat((speech_embeds, audio_embeds), dim=-1)
            speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)

            if self.window_level_Qformer:
                B, T, C = speech_embeds.shape
                kernel = round(1500 * self.second_per_window / 30.0)
                stride = round(1500 * self.second_stride / 30.0)
                kernel = (1, kernel)
                stride = (1, stride)
                speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
                speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
                _, _, L = speech_embeds_overlap.shape
                speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
                speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
                speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
                speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)

            query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
            query_output = self.speech_Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=speech_embeds,
                encoder_attention_mask=speech_atts,
                return_dict=True,
            )

            speech_embeds = self.speech_llama_proj(query_output.last_hidden_state)

            if self.window_level_Qformer:
                speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()

            speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)
        else:
            raise NotImplementedError

        return speech_embeds, speech_atts

    
    def adapt_transformer(self, speech_embeds, audio_embeds=None):
        # taken from SALMONN
        # with self.maybe_autocast():
        speech_embeds = self.ln_speech(speech_embeds)
        if audio_embeds is not None:
            audio_embeds = self.ln_audio(audio_embeds)
            if audio_embeds.size(1) < speech_embeds.size(1):
                audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
            elif audio_embeds.size(1) > speech_embeds.size(1):
                speech_embeds = F.pad(speech_embeds, (0, 0, 0, audio_embeds.size(1) - speech_embeds.size(1)))
            speech_embeds = torch.cat((speech_embeds, audio_embeds), dim=-1)
        speech_atts = torch.ones(speech_embeds.size()[1], speech_embeds.size()[1], dtype=torch.float).to(speech_embeds.device)

        transformer_output = self.transformer(src=speech_embeds, mask=speech_atts, src_key_padding_mask=None, is_causal=None)

        speech_embeds = self.speech_llama_proj(transformer_output)

        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)

        return speech_embeds, speech_atts


    
    def append_text_input(self, embeds, atts, prompt, multi_prompt=True):
        # print(prompt)
        if prompt:
            # if multi_prompt:
            p_before = []
            p_after = []
            for i, p in enumerate(prompt):
                b, a = p.split("<SpeechHere>")
                p_before.append(b)
                p_after.append(a)

            p_before_tokens = self.lm_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False
            ).to(embeds.device)
            p_before_embeds = self.language_model.model.embed_tokens(p_before_tokens.input_ids) if not self.lora else self.language_model.model.model.embed_tokens(p_before_tokens.input_ids)

            # speech_embeds wrapped with prompts_embeds are padded to the same length here
            p_after_tokens = self.lm_tokenizer(
                p_after, return_tensors="pt", padding="longest", add_special_tokens=False
            ).to(embeds.device)
            p_after_embeds = self.language_model.model.embed_tokens(p_after_tokens.input_ids) if not self.lora else self.language_model.model.model.embed_tokens(p_after_tokens.input_ids)

            wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
            wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)

            return wrapped_embeds, wrapped_atts
        else:
            return embeds, atts
    

    def append_text_input_mt(self, spec, prompt, multi_prompt=True):
   
        # tokens = self.lm_tokenizer(
        #     prompt, return_tensors="pt", add_special_tokens=False
        # ).to(embeds.device)
        # embeds = self.language_model.model.embed_tokens(tokens.input_ids) if not self.lora else self.language_model.model.model.embed_tokens(tokens.input_ids)
        # atts = tokens.attention_mask

        # return embeds, atts

        text = [str(t) + self.end_sym for t in prompt]
        mt_tokens = self.lm_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(spec.device)
        mt_embeds = self.language_model.model.embed_tokens(mt_tokens.input_ids) if not self.lora else self.language_model.model.model.embed_tokens(mt_tokens.input_ids)

        return mt_embeds, mt_tokens.attention_mask


    # def get target_embeds():

    
    
    def forward(self, spec, audio_array, audio_path, task, text_output, text_input, question_input, padding_mask):
        #  detect whether there are multi tasks in this batch
        # task = list(set(task))
        if len(task) > 1 or "QA" in task:
            self.multi_prompt = True

        # prepare prompts
        if self.prompt_dict:
            if self.multi_prompt:
                # print(task)
                prompt = [random.choice(self.prompt_dict[t]) for t in task]
                if question_input:
                    prompt = [p.format(q) if '{}' in p else p for p, q in zip(prompt, question_input) ]
                if text_input:
                    prompt = [p.format(q) if '{}' in p else p for p, q in zip(prompt, text_input) ]
            else:
                prompt = random.choice(self.prompt_dict[task][0])
        
        # not needed for mt
        # figure out how to do this within a batch?
        if not torch.all(torch.eq(audio_array, 0)):
            speech_embeds, audio_embeds = self.encode_speech_and_audio(spec, audio_array=audio_array, audio_array_paddded=audio_array, audio_padding_mask=padding_mask)

            # not needed for mt
            if self.config['model']['is_qformer']:
                speech_embeds, speech_atts = self.adapt_qformer(speech_embeds=speech_embeds, audio_embeds=audio_embeds)

            # not needed for mt
            if self.config['model']['is_transformer']:
                speech_embeds, speech_atts = self.adapt_transformer(speech_embeds=speech_embeds, audio_embeds=audio_embeds)

            if self.prompt_dict:
                # make different one for mt
                speech_embeds, speech_atts = self.append_text_input(speech_embeds, speech_atts, prompt, multi_prompt=self.multi_prompt)
        else:
            # different one for mt
            speech_embeds, speech_atts = self.append_text_input_mt(spec, prompt, multi_prompt=self.multi_prompt)


        # prepare inputs for LLM
        text = [str(t) + self.end_sym for t in text_output]
        to_regress_tokens = self.lm_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(spec.device)
        to_regress_embeds = self.language_model.model.embed_tokens(to_regress_tokens.input_ids) if not self.lora else self.language_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.lm_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones(
                [speech_atts.shape[0], speech_atts.shape[1] + 1],
                dtype=torch.long
            ).to(spec.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = speech_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1],
            dtype=to_regress_tokens.input_ids.dtype,
            device=to_regress_tokens.input_ids.device,
        ) * self.lm_tokenizer.bos_token_id
        bos_embeds = self.language_model.model.embed_tokens(bos) if not self.lora else self.language_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, speech_atts, to_regress_tokens.attention_mask], dim=1)

        # calulate loss
        # with self.maybe_autocast():
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )

        return outputs



    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        z = self(batch["spec"], batch["audio_array"], 
                batch["audio_path"], batch["task"], 
                batch["text_output"], batch["text_input"], batch["question_input"], 
                batch["padding_mask"])
        loss = z.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        bleu = BLEU()
        # this is the validation loop
        z = self(batch["spec"], batch["audio_array"], 
                 batch["audio_path"], batch["task"], 
                 batch["text_output"], batch["text_input"], batch["question_input"], 
                 batch["padding_mask"])
        val_loss = z.loss
        for i in range(0, len(batch['task'])):
            generation_tasks = self.config['dataset']['test_set_tasks']['generation_tasks']
            if any(elem in batch['task'][i] for elem in generation_tasks):
                text_out = self.generate(batch["spec"][i].unsqueeze(0), batch["audio_array"][i].unsqueeze(0), 
                        batch["audio_path"][i], batch["task"][i], 
                        batch["text_output"][i], batch["text_input"][i], batch["question_input"][i], 
                        batch["padding_mask"][i].unsqueeze(0), self.config, prompts=[batch['prompt'][i]])[0]


                if 'asr' in batch['task'][i]:
                    self.wer_refs.append(batch["text_output"][i])
                    self.wer_hyps.append(text_out[0])
                    wer_calc_ind = jiwer.wer(batch["text_output"][i], text_out[0])
                    self.wer_ind.append(wer_calc_ind)

                if 'translation' in batch['task'][i]:
                    self.bleu_syss.append(text_out[0])
                    self.bleu_refs.append(batch["text_output"][i])
                    b_score_ind = bleu.corpus_score([text_out[0]], [[batch["text_output"][i]]]).score
                    self.bleu_ind.append(b_score_ind)

                # if QA, get F1 or accuracy

            likelihood_based_tasks = self.config['dataset']['test_set_tasks']['likelihood_based_tasks']
            if any(elem in batch['task'][i] for elem in likelihood_based_tasks):
                if 'lang_id' in batch['task'][i]:
                    selected_option = self.get_like(batch["spec"][i].unsqueeze(0), batch["audio_array"][i].unsqueeze(0), 
                        batch["audio_path"][i], [batch["task"][i]], 
                        batch["text_output"][i], batch["question_input"][i], 
                        batch["padding_mask"][i].unsqueeze(0), self.config, self.lang_id_options, prompts=[batch['prompt'][i]])

                    self.lang_truth.append(batch["text_output"][i])
                    self.lang_selected.append(selected_option)

                    if batch["text_output"][i] in selected_option.lower():
                        self.lang_id_acc.append(1)
                    else:
                        self.lang_id_acc.append(0)
                    
        if self.wer_refs:
            err = jiwer.wer(self.wer_refs, self.wer_hyps)
            self.log("val_wer", torch.tensor(err, dtype=torch.float16))
            data = list(zip(self.wer_refs, self.wer_hyps, self.wer_ind))
            for d in data:
                self.asr_table.add_data(d[0], d[1], d[2])

        if self.bleu_syss:
            b_score = bleu.corpus_score(self.bleu_syss, [self.bleu_refs]).score
            self.log("val_bleu", torch.tensor(b_score, dtype=torch.float16))
            data = list(zip(self.bleu_refs, self.bleu_syss, self.bleu_ind))
            for d in data:
                self.ast_table.add_data(d[0], d[1], d[2])

        if self.lang_id_acc:
            lid_acc = np.mean(self.lang_id_acc)
            self.log("val_lang_id_acc", torch.tensor(lid_acc, dtype=torch.float16))
            data = list(zip(self.lang_truth, self.lang_selected, self.lang_id_acc))
            for d in data:
                self.lid_table.add_data(d[0], d[1], d[2])

        self.log('val_loss', val_loss, sync_dist=True)

    
    def test_step(self, batch, batch_idx):
        bleu = BLEU()

        z = self(batch["spec"], batch["audio_array"], 
                 batch["audio_path"], batch["task"], 
                 batch["text_output"], batch["question_input"], 
                 batch["padding_mask"])
        test_loss = z.loss

        for i in range(0, len(batch['task'])):
            generation_tasks = ['asr', 'QA', 'translation']
            if any(elem in batch['task'][i] for elem in generation_tasks):
                text_out = self.generate(batch["spec"][i].unsqueeze(0), batch["audio_array"][i].unsqueeze(0), 
                        batch["audio_path"][i], batch["task"][i], 
                        batch["text_output"][i], batch["text_input"][i], batch["question_input"][i], 
                        batch["padding_mask"][i].unsqueeze(0), self.config, prompts=[batch['prompt'][i]])[0]

                if 'asr' in batch['task'][i]:
                    self.wer_refs.append(batch["text_output"][i])
                    self.wer_hyps.append(text_out[0])
                    wer_calc_ind = jiwer.wer(batch["text_output"][i], text_out[0])
                    self.wer_ind.append(wer_calc_ind)

                if 'translation' in batch['task'][i]:
                    self.bleu_syss.append(text_out[0])
                    self.bleu_refs.append(batch["text_output"][i])
                    b_score_ind = bleu.corpus_score([text_out[0]], [[batch["text_output"][i]]]).score
                    self.bleu_ind.append(b_score_ind)

                # if QA, get F1 or accuracy

            likelihood_based_tasks = ['lang_id']
            if any(elem in batch['task'][i] for elem in likelihood_based_tasks):
                if 'lang_id' in batch['task'][i]:
                    selected_option = self.get_like(batch["spec"][i].unsqueeze(0), batch["audio_array"][i].unsqueeze(0), 
                        batch["audio_path"][i], [batch["task"][i]], 
                        batch["text_output"][i], batch["question_input"][i], 
                        batch["padding_mask"][i].unsqueeze(0), self.config, self.lang_id_options, prompts=[batch['prompt'][i]])

                    self.lang_truth.append(batch["text_output"][i])
                    self.lang_selected.append(selected_option)

                    if batch["text_output"][i] in selected_option.lower():
                        self.lang_id_acc.append(1)
                    else:
                        self.lang_id_acc.append(0)
                    
        if self.wer_refs:
            err = jiwer.wer(self.wer_refs, self.wer_hyps)
            self.log("test_wer", torch.tensor(err, dtype=torch.float16))
            data = list(zip(self.wer_refs, self.wer_hyps, self.wer_ind))
            for d in data:
                self.asr_table.add_data(d[0], d[1], d[2])

        if self.bleu_syss:
            b_score = bleu.corpus_score(self.bleu_syss, [self.bleu_refs]).score
            self.log("test_bleu", torch.tensor(b_score, dtype=torch.float16))
            data = list(zip(self.bleu_refs, self.bleu_syss, self.bleu_ind))
            for d in data:
                self.ast_table.add_data(d[0], d[1], d[2])

        if self.lang_id_acc:
            lid_acc = np.mean(self.lang_id_acc)
            self.log("test_lang_id_acc", torch.tensor(lid_acc, dtype=torch.float16))
            data = list(zip(self.lang_truth, self.lang_selected, self.lang_id_acc))
            for d in data:
                self.lid_table.add_data(d[0], d[1], d[2])

        self.log("test_loss", test_loss)


    def on_validation_epoch_end(self):
        # self.lid_artifact = wandb.Artifact(name=wandb.run.name + "_" + f"lid_epoch{self.current_epoch}", type=wandb.run.name)
        # self.lid_artifact.add(self.lid_table, f"lid_epoch{self.current_epoch}")
        # self.logger.experiment.log_artifact(self.lid_artifact, aliases=[f'step_{self.global_step}', f'epoch_{self.current_epoch}'])
        # self.logger.experiment.log({"Lang ID": self.lid_table})
        # self.lang_id_acc = []
        # self.lang_truth = []
        # self.lang_selected = []
        # self.lid_table = wandb.Table(columns=self.langid_columns)


        """
        self.asr_artifact = wandb.Artifact(name=wandb.run.name + "_" + f"asr_epoch{self.current_epoch}", type=wandb.run.name)
        self.asr_artifact.add(self.asr_table, f"asr_epoch{self.current_epoch}")
        self.logger.experiment.log_artifact(self.asr_artifact, aliases=[f'step_{self.global_step}', f'epoch_{self.current_epoch}'])
        self.logger.experiment.log({"ASR": self.asr_table})
        self.wer_refs = []
        self.wer_hyps = []
        self.wer_ind = []
        self.asr_table = wandb.Table(columns=self.asr_columns)
        """

        # self.ast_artifact = wandb.Artifact(name=wandb.run.name + "_" + f"ast_epoch{self.current_epoch}", type=wandb.run.name)
        # self.ast_artifact.add(self.asr_table, f"ast_epoch{self.current_epoch}")
        # self.logger.experiment.log_artifact(self.ast_artifact, aliases=[f'step_{self.global_step}', f'epoch_{self.current_epoch}'])
        # self.logger.experiment.log({"AST": self.ast_table})
        # self.bleu_syss = []
        # self.bleu_refs = []
        # self.bleu_ind = []
        # self.ast_table = wandb.Table(columns=self.ast_columns)


    def configure_optimizers(self):
        
        beta2 = self.config['lightningargs']['optims'].get("beta2", 0.999)


        optimizer = optim.AdamW(self.parameters(), 
                                lr=float(self.config['lightningargs']['optims']['init_lr']),
                                weight_decay=float(self.config['lightningargs']['optims']['weight_decay']),
                                betas=(0.9, beta2)
                                )

        lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, 
                                                num_warmup_steps=self.config['lightningargs']['optims']['warmup_steps'], 
                                                num_training_steps=self.num_training_steps) 

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'step','frequency': 1}}
        # return optimizer
    # def configure_optimizers(self):
    #     return optim.AdamW(self.parameters(), lr=0.001)

    def generate(self, spec, audio_array, audio_path, task, text_output, text_input, question_input, padding_mask, config, prompts=None):
        batch_size = spec.shape[0]

        if not torch.all(torch.eq(audio_array, 0)):
            speech_embeds, audio_embeds = self.encode_speech_and_audio(spec.to('cuda'), audio_array=audio_array.to('cuda'), audio_array_paddded=audio_array, audio_padding_mask=padding_mask.to('cuda'))

            # not needed for mt
            if self.config['model']['is_qformer']:
                speech_embeds, speech_atts = self.adapt_qformer(speech_embeds=speech_embeds, audio_embeds=audio_embeds)

            # not needed for mt
            if self.config['model']['is_transformer']:
                speech_embeds, speech_atts = self.adapt_transformer(speech_embeds=speech_embeds, audio_embeds=audio_embeds)

            if prompts is not None:
                # make different one for mt
                speech_embeds, speech_atts = self.append_text_input(speech_embeds, speech_atts, prompts, multi_prompt=self.multi_prompt)
        else:
            # different one for mt
            speech_embeds, speech_atts = self.append_text_input_mt(spec, prompts, multi_prompt=self.multi_prompt)

        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.int32,
            device=speech_embeds.device,
        ) * self.lm_tokenizer.bos_token_id
        bos_embeds = self.language_model.model.embed_tokens(bos) if not self.lora else self.language_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        embeds = torch.cat([bos_embeds, speech_embeds], dim=1).half()
        attns = torch.cat([atts_bos, speech_atts], dim=1).half()

        terminators = [self.lm_tokenizer.eos_token_id, self.lm_tokenizer.convert_tokens_to_ids(config['prompts']['end_sym'])]
        outputs = self.language_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=config['generation']["max_new_tokens"],
            num_beams=config['generation']["num_beams"],
            do_sample=config['generation']["do_sample"],
            eos_token_id=terminators,
            min_length=config['generation']["min_length"],
            temperature=config['generation']["temperature"],
            top_p=config['generation']["top_p"],
            repetition_penalty=config['generation']['repetition_penalty'],
            length_penalty=config['generation']["length_penalty"],
            attention_mask=attns,
        )
        text = self.lm_tokenizer.batch_decode(outputs, add_special_tokens=False)
        # print(text[0])
        # text = text[0].split('<EOS_TOKEN>')[0]
        # text = text[0].split(config['prompts']['end_sym'])[0]
        # print(text)

        return text


    def get_like(self, spec, audio_array, audio_path, task, text_output, text_input, question_input, padding_mask, config, options, prompts=None,):
        
        if len(task) > 1 or "QA" in task:
            self.multi_prompt = True

        # prepare prompts
        if self.prompt_dict:
            if self.multi_prompt:
                prompt = [random.choice(self.prompt_dict[t]) for t in task]
                if question_input:
                    prompt = [p.format(q) if '{}' in p else p for p, q in zip(prompt, question_input) ]
            else:
                prompt = random.choice(self.prompt_dict[task][0])

        speech_embeds, audio_embeds = self.encode_speech_and_audio(spec, audio_array=audio_array, audio_array_paddded=audio_array, audio_padding_mask=padding_mask)

        speech_embeds, speech_atts = self.adapt_qformer(speech_embeds=speech_embeds, audio_embeds=audio_embeds)

        if self.prompt_dict:
            speech_embeds, speech_atts = self.append_text_input(speech_embeds, speech_atts, prompt, multi_prompt=self.multi_prompt)


        op_loss = []
        op_loss_dict = {}
        # prepare inputs for LLM
        for op in options:
            text = op + self.end_sym
            to_regress_tokens = self.lm_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(spec.device)
            to_regress_embeds = self.language_model.model.embed_tokens(to_regress_tokens.input_ids) if not self.lora else self.language_model.model.model.embed_tokens(to_regress_tokens.input_ids)
            targets = to_regress_tokens.input_ids.masked_fill(
                to_regress_tokens.input_ids == self.lm_tokenizer.pad_token_id, -100
            )
            empty_targets = (
                torch.ones(
                    [speech_atts.shape[0], speech_atts.shape[1] + 1],
                    dtype=torch.long
                ).to(spec.device).fill_(-100)
            )
            targets = torch.cat([empty_targets, targets], dim=1)

            batch_size = speech_embeds.shape[0]
            bos = torch.ones(
                [batch_size, 1],
                dtype=to_regress_tokens.input_ids.dtype,
                device=to_regress_tokens.input_ids.device,
            ) * self.lm_tokenizer.bos_token_id
            bos_embeds = self.language_model.model.embed_tokens(bos) if not self.lora else self.language_model.model.model.embed_tokens(bos)
            atts_bos = speech_atts[:, :1]

            inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds], dim=1)
            attention_mask = torch.cat([atts_bos, speech_atts, to_regress_tokens.attention_mask], dim=1)

            # calulate loss
            # with self.maybe_autocast():
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

                # op_loss.append({op: loss.item()})
            op_loss_dict[op] = loss.item()


        sorted_op_loss = dict(sorted(op_loss_dict.items(), key=lambda item: item[1]))

        return list(sorted_op_loss.keys())[0]


    @property
    def num_training_steps(self) -> int:
        """Get number of training steps"""
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps

        self.trainer.fit_loop.setup_data()
        dataset_size = len(self.trainer.train_dataloader)
        num_steps = dataset_size * self.trainer.max_epochs

        return num_steps

