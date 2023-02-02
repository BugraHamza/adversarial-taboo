import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModel


class ClassifyBERTurk(nn.Module):
    def __init__(self, config):
        super(ClassifyBERTurk, self).__init__()
        self.berturk = AutoModel.from_config(config)
        self.fc = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, **inputs):
        bert_out = self.berturk(**inputs)
        bert_last_hidden = bert_out[0][:, 0, :]
        
        fc_out = self.fc(bert_last_hidden)
        sigmoid_out = self.sigmoid(fc_out)
        return sigmoid_out


def get_fluency_model(model_name_or_path, tokenizer_length=None, device='cpu'):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device=device)

    if tokenizer_length:
        model.resize_token_embeddings(tokenizer_length)

    return model
