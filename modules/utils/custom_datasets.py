import torch
from torch.utils.data import Dataset


def tokenize(text, tokenizer):
    if tokenizer.bos_token is not None:
        text = tokenizer.bos_token + text + tokenizer.eos_token

    return tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')


class FluencyDataset(Dataset):
    def __init__(self, data, tokenizer, device='cpu'):
        super().__init__()

        self.data = data
        self.tokenizer = tokenizer
        self.device = device

    def __getitem__(self, i):
        text = self.data.content[i]
        tokenized_text = tokenize(text, self.tokenizer)

        return {k: v.squeeze().to(self.device) for k, v in tokenized_text.items()}

    def __len__(self):
        return len(self.data)


class RelevancyDataset(Dataset):
    def __init__(self, data, tokenizer, device='cpu'):
        super().__init__()

        self.data = data
        self.tokenizer = tokenizer
        self.device = device

    def __getitem__(self, i):
        row = self.data.iloc[i]
        post, resp, y = row['post'], row['response'], row['is_pair']

        tokenized_pair = self.tokenizer(post, resp, padding='max_length', truncation='longest_first', return_tensors='pt')
        tokenized_pair = {k: v.squeeze().to(self.device) for k, v in tokenized_pair.items()}

        return tokenized_pair, torch.tensor(y, dtype=torch.float).to(self.device)

    def __len__(self):
        return len(self.data)
