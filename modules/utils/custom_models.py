import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModel


class ClassifyBERTurk(nn.Module):
    def __init__(self, model_name_or_path):
        super(ClassifyBERTurk, self).__init__()

        self.berturk = AutoModel.from_pretrained(model_name_or_path)
        self.fc = nn.Linear(self.berturk.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, **inputs):
        bert_out = self.berturk(**inputs).last_hidden_state
        bert_last_hidden = bert_out[:, 0, :]
        
        fc_out = self.fc(bert_last_hidden)
        sigmoid_out = self.sigmoid(fc_out)
        return sigmoid_out


def get_fluency_model(model_name_or_path, tokenizer_length=None, device='cpu'):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device=device)

    if tokenizer_length:
        model.resize_token_embeddings(tokenizer_length)

    return model


def get_relevancy_model(model_name_or_path, trainable_llm=False, device='cpu'):
    model = ClassifyBERTurk(model_name_or_path).to(device=device)
    print('Trainable LLM:', trainable_llm)
    for params in model.berturk.parameters():
        params.requires_grad = trainable_llm

    return model
