from torch.utils.data import Dataset


def tokenize(text, tokenizer):
    if tokenizer.bos_token is not None:
        text = tokenizer.bos_token + text + tokenizer.eos_token

    return tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')


class FluencyDataset(Dataset):
    def __init__(self, data, tokenizer, device='cpu'):
        self.data = data
        self.tokenizer = tokenizer
        self.device = device

    def __getitem__(self, i):
        text = self.data.content[i]
        tokenized_text = tokenize(text, self.tokenizer)

        return {k: v.squeeze().to(self.device) for k, v in tokenized_text.items()}

    def __len__(self):
        return len(self.data)


class AdequacyDataset(Dataset):
    def __init__(self, data, tokenizer, device='cpu'):
        self.data = data
        self.tokenizer = tokenizer
        self.device = device

    def __getitem__(self, i):
        row = self.data.iloc[i]
        post, resp = row['post'], row['response']

        el_sent = row.post + self.tokenizer.sep_token + row.response
        tokenized_el = tokenize(el_sent, self.tokenizer)

        return {k: v.squeeze().to(self.device) for k, v in tokenized_el.items()}

    def __len__(self):
        return len(self.data)
