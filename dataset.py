from torch.utils.data import Dataset
import pandas as pd
import soundfile as sf
from utils import batch_audio_array, tok_labels, check_file_exists, get_valid_files_df
from transformers import DataCollatorWithPadding, EvalPrediction, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import typing
import torch
import json
from io import StringIO
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from pathlib import Path

pd.options.mode.chained_assignment = None  # default='warn'




class SpeechLMDataset(Dataset):
    def __init__(self, config, name_or_path, limit_dev_dataset=0):
        super().__init__()

        # self.path_label = 'path'
        self.path_label = config['dataset']['labels'].get('path_label', 'path')
        self.audio_label = 'audio'
        # self.text_output_label = 'text'
        self.text_output_label = config['dataset']['labels'].get('text_output_label', 'text')
        self.question_input_label = config['dataset']['labels'].get('question_input_label', 'Q')
        self.task_label = config['dataset']['labels'].get('task_label', 'task')
        self.text_input_label = config['dataset']['labels'].get('text_input_label', 'input')
        self.prompt_label = 'Prompt'
        self.config = config
        self.limit_dev_dataset = limit_dev_dataset

        if config['dataset']['is_hf']:
            self.ds = load_dataset(self.config['dataset']['hf_dataset_name_or_path'])
            self.ds_type = 'hf'

        if self.config['dataset']['is_json']:
            j = json.load(open(Path(name_or_path), "r"))["annotation"]
            df = pd.DataFrame.from_records(j)
            self.ds_type = 'json'
            self.ds = df
            
            # self.ds = get_valid_files_df(new_df, self.path_label, name_or_path)

        if self.config['dataset']['is_jsonl']:
            df = pd.read_json(Path(name_or_path), lines=True)
            # self.ds = self.filter_dataset(df)
            self.ds = df

        if self.config['dataset']['is_tsv']:
            df = pd.read_csv(name_or_path, sep='\t')
            # self.ds = df
            self.ds = self.filter_dataset(df)
            self.ds_type = 'tsv'

        if self.limit_dev_dataset > 0:
            samples_per_task = self.limit_dev_dataset 
            new_ds = pd.DataFrame(columns=self.ds.columns)

            for t in self.ds[self.task_label].unique():
                task_df = self.ds[self.ds[self.task_label] == t]
                task_df_samp = task_df.sample(n=samples_per_task, random_state=43)
                new_ds = pd.concat([new_ds, task_df_samp], ignore_index=True)

            self.ds = new_ds

        with open(self.config['prompts']['test_prompts_path']) as j_data:
            self.prompt_dict = json.load(j_data)



    def __len__(self):

        return len(self.ds)



    def __getitem__(self, index):

        # elem = self.ds[index]
        elem = self.ds.iloc[index]

        audio_path = elem.get(self.path_label, None)
        if audio_path:
            data, samplerate = sf.read(audio_path)
            elem[self.audio_label] = {"array": data, "array_padded": data, "sampling_rate": samplerate}
        else:
            elem[self.audio_label] = None

        # if not elem[self.question_input_label]:
        #     elem[self.question_input_label] = ""
        question_input = elem.get(self.question_input_label, "")
       
        # if elem[self.path_label]:
        #     data, samplerate = sf.read(elem[self.path_label])
        #     # data, samplerate = torchaudio.load(elem[self.path_label],format='wav')
        #     elem[self.audio_label] = {"array": data, "array_padded": data, "sampling_rate": samplerate}
        # else:
        #     elem[self.path_label] = None
        #     elem[self.audio_label] = None
        
        text_input = elem.get(self.text_input_label, "")
        text_output = elem.get(self.text_output_label, "")
        # if not elem[self.text_input_label]:
        #     elem[self.text_inut_label] = ""p

        audio_dict = elem.get(self.audio_label, None)
        if audio_dict:
            audio_array = audio_dict['array']
        else:
            audio_array = np.zeros(16000)


        return { 'audio_array': audio_array,
                 'task': elem[self.task_label],
                 'text_input': text_input,
                 'text_output': text_output,
                 'question_input': question_input,
                 'audio_path': audio_path,
                 'padding_mask':audio_array,
                 'prompt': self.prompt_dict[elem[self.task_label]],
       }


    def filter_dataset(self, ds, len):

        # for i, r in tqdm(ds.iterrows(), total=ds.shape[0]):
        #     data, samplerate = sf.read(r['path'])
        #     len_f = len(data) / samplerate

        #     if len_f > 15:
        #         ds = ds.drop(i)

        ds = ds[ds['len'] < 29]
        # print(ds)

        

        return ds
             





class SpeechLMCollator(DataCollatorWithPadding):

    def __call__(self, features: typing.List[typing.Dict[str, typing.Union[typing.List[int], torch.Tensor]]]) -> typing.Dict[str, torch.Tensor]:
        # if audio_array:
        audio_array = [torch.from_numpy(feature["audio_array"]) for feature in features]
        # audio_array = [feature["audio_array"] for feature in features]
        # label_features = [{"input_ids": feature["labels"]} for feature in features]
        # raw_wav_length = torch.tensor([len(feat["raw_wav"]) for s in samples])
        audio_array_len = torch.tensor([len(feature["audio_array"]) for feature in features])
        raw_wav = pad_sequence(audio_array, batch_first=True, padding_value=0)
        paddding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= audio_array_len.unsqueeze(1)

        # print(paddding_mask)
        # print(audio_array)
        # print(audio_array_len)

        batch = {}
        batch['spec'] = torch.stack([self.tokenizer(audio['audio_array'], sampling_rate=16000, return_tensors="pt")["input_features"].squeeze() for audio in features], dim=0)
        batch["audio_array"] = raw_wav
        batch["padding_mask"] = paddding_mask
        batch["text_output"] = [feature['text_output']for feature in features]
        batch["text_input"] = [feature['text_input']for feature in features]
        batch["task"] = [feature['task']for feature in features]
        batch['question_input'] = [feature['question_input']for feature in features]
        batch['audio_path'] = [feature['audio_path']for feature in features]
        batch['prompt'] = [feature['prompt']for feature in features]

        # print(batch)

        return batch



    def make_individual(self, path):
            data, samplerate = sf.read(path)
            audio_array = [torch.from_numpy(data)]
            # audio_array = [feature["audio_array"] for feature in features]
            # label_features = [{"input_ids": feature["labels"]} for feature in features]
            # raw_wav_length = torch.tensor([len(feat["raw_wav"]) for s in samples])
            audio_array_len = torch.tensor([len(data)])
            raw_wav = pad_sequence(audio_array, batch_first=True, padding_value=0)
            paddding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= audio_array_len.unsqueeze(1)

            # print(paddding_mask)
            # print(audio_array)
            # print(audio_array_len)

            # print(self.tokenizer(data, sampling_rate=16000, return_tensors="pt"))

            batch = {}
            batch['spec'] = self.tokenizer(data, sampling_rate=16000, return_tensors="pt")["input_features"]
            batch["audio_array"] = raw_wav
            batch["padding_mask"] = paddding_mask
            batch["text_output"] = ""
            batch["text_input"] = ""
            batch["task"] = ""
            batch['question_input'] = ""
            batch['audio_path'] = path

            return batch