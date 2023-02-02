import torch
from modules.utils import *


class JudgeSystem:
    def __init__(self, fluency_path, relevancy_path, fluency_threshold=35e3, relevancy_threshold=0.4):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.fluency_model = torch.load(fluency_path, map_location=self.device)
        self.fluency_tokenizer = get_gpt_tokenizer(self.fluency_model.config.name_or_path)
        self.fluency_threshold = fluency_threshold

        self.relevancy_model = torch.load(relevancy_path, map_location=self.device)
        self.relevancy_tokenizer = get_bert_tokenizer(self.relevancy_model.berturk.config.name_or_path)
        self.relevancy_threshold = relevancy_threshold

    def check_perplexity(self, post, response):
        sep_token = self.fluency_tokenizer.special_tokens_map['sep_token']
        sentence = ' '.join([post, sep_token, response]) if response else post
        perplexity = calc_perplexity(self.fluency_model, self.fluency_tokenizer, sentence)

        # LOGGING
        # print(f'[LOG]: Perplexity: {perplexity}')

        return perplexity < self.fluency_threshold

    def check_relevancy(self, post, response):
        sep_token = self.relevancy_tokenizer.special_tokens_map['sep_token']
        sentence = ' '.join([post, sep_token, response]) if response else post
        tokenized_sent = self.relevancy_tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
        loss = self.relevancy_model(**tokenized_sent).item()

        # LOGGING
        # print(f'[LOG]: Relevancy: {loss}')

        return loss > self.relevancy_threshold

    def __call__(self, curr_sent, prev_sent=None):
        perplexity = self.check_perplexity(curr_sent, prev_sent)
        relevancy = self.check_relevancy(curr_sent, prev_sent)

        return perplexity and relevancy


if __name__ == '__main__':
    judge = JudgeSystem(r'../../saved-models/fluency_model_3.1080100260537895', 'adequacy_model_0.5374999642372131')

    sent1 = "Bu ben yalan haahsjasdksdahkjas ayrımcılık para ışık neden leyla. ç  ğiü hatıra tokat jandarma "
    ans1 = "Soruyor sen"
    print(sent1)
    print(judge(curr_sent=ans1, prev_sent=sent1))
    print()
