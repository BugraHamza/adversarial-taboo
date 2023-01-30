import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, X, y, tokenizer, device='cpu'):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.device = device
        
    def __getitem__(self, i):
        post, resp = X[i]        
        el_sent = ' '.join([post, self.tokenizer.special_tokens_map['sep_token'], resp])
        tokenized_el = self.tokenizer(el_sent, padding='max_length', return_tensors='pt')
        
        for k, v in tokenized_el.items():
            v_device = v.to(self.device)               
            tokenized_el[k] = v_device.squeeze()
        
        return tokenized_el, self.y[i]
    
    def __len__(self):
        return len(self.y)
    