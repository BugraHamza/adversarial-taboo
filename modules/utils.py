import math
import random

import requests

import torch
from transformers import GPT2Tokenizer, BertTokenizer


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

    for k, v in more_tokens_dict.items():
        special_tokens_dict[k] = v
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    
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
    # open the file and get one hop concepts for each word
    with open('/Users/bugrahamzagundog/Desktop/Courses/AutoTaboo-Player/modules/Word Selection/selected_words_mini.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            word = line.strip()
            concepts = get_one_hop_concepts(word)
            if concepts:
                print(word)