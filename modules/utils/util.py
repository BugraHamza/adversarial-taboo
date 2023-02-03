from __future__ import annotations

import re

import math
import random

import pandas as pd
import requests

import torch
from transformers import GPT2Tokenizer, BertTokenizer

from sklearn.model_selection import train_test_split

from typing import Union, Dict, List


def split_data(data, sizes: Union[Dict | List], random_state=42):
    if isinstance(sizes, dict):
        sizes = [sizes['train'], sizes['val'], sizes['test']]

    if sum(sizes) != 1.0:
        sizes = [s / sum(sizes) for s in sizes]

    train_val, test = train_test_split(data, test_size=sizes[2], random_state=random_state)
    train, val = train_test_split(train_val, test_size=sizes[1], random_state=random_state)

    return train, val, test


def prepare_data_for_modeling(data):
    if 'content' not in data.columns:
        data = pd.concat([data.post, data.response], ignore_index=True)

    data = data.apply(lambda x: re.sub(r'\s', ' ', x))
    data = data.apply(lambda x: re.sub('View Poll', '', x))
    data = data.str.strip()

    return data.tolist()


def prepare_data_for_pairs(data):
    pass


def calc_perplexity(model, tokenizer, sentence, device='cpu'):
    sent = tokenizer.bos_token + sentence + tokenizer.eos_token
    sent = tokenizer(sent, return_tensors='pt', truncation=True, padding=True)
    sent = sent['input_ids'].to(device=device)
    
    model.eval()
    with torch.no_grad():
        loss = model(sent, labels=sent).loss
        perplexity = math.exp(loss)
        
        return perplexity


def get_gpt_tokenizer(path, max_len=512, more_tokens_dict={}):
    tokenizer = GPT2Tokenizer.from_pretrained(path, model_max_length=max_len)
    special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>',
                           'pad_token': '<PAD>', 'sep_token': '<SEP>'}

    special_tokens_dict = {**special_tokens_dict, **more_tokens_dict}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    return tokenizer


def get_bert_tokenizer(path):
    tokenizer = BertTokenizer.from_pretrained(path)
    return tokenizer


def concept_generator(word):
    url = 'http://api.conceptnet.io/c/tr/' + word + '?rel=/r/RelatedTo&limit=1000'
    response = requests.get(url)
    data = response.json()

    # get one hop words
    one_hop_list = set()
    for concept in data['edges']:
        if concept['start']['language'] == 'tr':
            one_hop_word = concept['start']['label'].lower()
            if word not in one_hop_word:
                one_hop_list.add(one_hop_word)

    yield random.choice(list(one_hop_list))


if __name__ == '__main__':
    # read and prepare data for language modeling task
    data = pd.read_parquet('/Users/quimba/Desktop/adversarial-taboo/datasets/reddit-dataset/tr-reddit.parquet')
    data = prepare_data_for_modeling(data)

    # split data into train, validation, and test sets
    train_data, val_data, test_data = split_data(data, {'train': 0.8, 'val': 0.1, 'test': 0.1})

    pd.DataFrame({'content': train_data}).to_parquet('/Users/quimba/Desktop/adversarial-taboo/datasets/reddit-dataset/tr-reddit_train.parquet')
    pd.DataFrame({'content': val_data}).to_parquet('/Users/quimba/Desktop/adversarial-taboo/datasets/reddit-dataset/tr-reddit_val.parquet')
    pd.DataFrame({'content': test_data}).to_parquet('/Users/quimba/Desktop/adversarial-taboo/datasets/reddit-dataset/tr-reddit_test.parquet')
