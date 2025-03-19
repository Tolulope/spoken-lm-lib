import datasets as hfds
import jiwer
import omegaconf as oc
import pandas as pd
import numpy as np
import soundfile as sf
import os
import wandb
import json
import re
from datasets import Dataset, Audio
from transformers import AutoTokenizer,StoppingCriteria
import torch
from tqdm import tqdm

from lightning.pytorch.loggers import WandbLogger

import io
import wandb
import base64
from PIL import Image

import soundfile as sf
from tqdm import tqdm


def announce(announcement):
    total_width = os.get_terminal_size().columns
    pad_length  = int(total_width/2 - len(announcement)/2 - 1)

    print(f"{'-' * pad_length} {announcement} {'-' * pad_length}")

def make_config(config):

    # Overwrite config vars with anything supplied in the command line
    config = oc.OmegaConf.merge(
        oc.OmegaConf.load(config['--config']),
        oc.OmegaConf.from_cli()
    )

    flat_args_long = pd.json_normalize(oc.OmegaConf.to_container(config), sep=".").melt(var_name='argument')
    missing_args   = flat_args_long.query("value == '???'")

    assert len(missing_args) == 0, f"""
    
    The following required arguments are missing:
    
        {','.join(missing_args['argument'].to_list())}

    """

    announce("Configuring environment")

    # Set environment variables
    for key, value in config['env'].items():

        if key == 'CUDA_VISIBLE_DEVICES':
            # OmegaConf will coerce number-like values into integers
            # but CUDA_VISIBLE_DEVICES should be a (comma-seperated) string
            value = str(value)

        os.environ[key] = value

    # Save a copy of final config in output_dir
    os.makedirs(config['lightningargs']['output_dir'], exist_ok=True)
    pos_config = os.path.join(config['lightningargs']['output_dir'], 'train_config.yaml')

    oc.OmegaConf.save(config, pos_config)
        
    if not 'wandb' in config.keys():

        return config, None

    else:
        wandb.login()
        
        wandb_logger = WandbLogger(allow_val_change=True, save_dir=config['lightningargs']['output_dir'], **config['wandb'])

        # if config.get("--run_name"):
        #     # Interpolate 'lr={tranargs[learning_rate]}' to 'lr=0.0001', where config['tranargs']['learning_rate'] = 0.0001
        #     run.name = config["--run_name"].format(**config)

        # # Log hyper-parameters not automatically tracked by wandb
        # untracked_args = flat_args_long[ ~flat_args_long.argument.str.contains("w2v2|trainargs|wandb|--", regex=True) ]
        # # Convert to flat dict, e.g. { 'data.base_path' : '/path/to/the/data' }
        # untracked_args = dict([ (d['argument'], d['value']) for d in untracked_args.to_dict(orient='records') ])

        # wandb.config.update(untracked_args, allow_val_change=True)

        # config['trainargs']['report_to'] = "wandb"

        return config, wandb_logger

def check_file_exists(filepath):
    return os.path.exists(filepath)


def get_valid_files_df(df, path_label, name_or_path):
    announce("Checking sound files")
    verified_df = pd.DataFrame(columns=df.columns)
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            sf.read(row[path_label])
            verified_df.loc[len(verified_df)] = row
        except RuntimeError:
            pass

    verified_df.to_csv(name_or_path.split('/')[-1].split('.')[0] + '.tsv', sep='\t', index=False)
    
    return verified_df


def batch_audio_array(row):

    data, samplerate = sf.read(row['path'])

    return {"array": data, "array_padded": data, "sampling_rate": samplerate}


def tok_labels(lab, config):
    tok =  AutoTokenizer.from_pretrained(config['model']['lm_name_or_path'])
    ids = tok.encode(row['text_output'])

    return ids


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


def wandb_htmltable(data, columns, css=''):
        # Helper functions
        def check_shape():
            num_cols = len(columns)
            for i,row in enumerate(data):
                if len(row) != num_cols:
                    raise ValueError(f'Row {i} in data should have {num_cols} cols, but found to have {len(row)}.')
        def join(list_str):
            return "".join(list_str)
        def wrap_html_tag(tag, content):
            html_content = content
            inline_style = ''
            if type(content) == dict:
                html_content = content['content']
                inline_style = content.get('style', inline_style)
            if Image.isImageType(html_content):	# if PIL image
                html_content = format_image(html_content, inline_style)
            return f'<{tag} style="{inline_style}">{html_content}</{tag}>'
        def format_image(pil_image, inline_style):
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            btyestr = base64.b64encode(img_byte_arr.getvalue()).decode()
            return f'<img style="{inline_style}" src="data:image/png;base64,{btyestr}" />'
        def format_html_tr(list_data, is_header=False):
            return wrap_html_tag('tr', join([format_html_td(cd, is_header) for cd in list_data]))
        def format_html_td(cell_data, is_header):
            return wrap_html_tag('th' if is_header else 'td', cell_data)

        check_shape()
        if not css:
            css = wrap_html_tag('style', '''
            table, th, td {
                border: 1px solid black;
                border-collapse: collapse;
            }''')
        column_html = format_html_tr(columns, is_header=True)
        rows_html = join([format_html_tr(row) for row in data])
        table_html = wrap_html_tag('table', join([column_html, rows_html]))
        body_html = wrap_html_tag('body', table_html)
        head_html = wrap_html_tag('head', css)
        iframe_html = wrap_html_tag('html', join([head_html, body_html]))
        return wandb.Html(iframe_html, inject=True)