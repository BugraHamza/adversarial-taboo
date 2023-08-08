import torch
from modules.utils.util import get_gpt_tokenizer, get_bert_tokenizer, calc_perplexity
from transformers import GPT2LMHeadModel


class JudgeSystem:
    def __init__(self, fluency_path, relevancy_path, fluency_threshold=75, relevancy_threshold=0.3752115408021162):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.fluency_model = GPT2LMHeadModel.from_pretrained(fluency_path).to(self.device)
        self.fluency_tokenizer = get_gpt_tokenizer('redrussianarmy/gpt2-turkish-cased')
        self.fluency_threshold = fluency_threshold

        self.relevancy_model = torch.load(relevancy_path, map_location=self.device)
        self.relevancy_tokenizer = get_bert_tokenizer('dbmdz/bert-base-turkish-cased')
        self.relevancy_threshold = relevancy_threshold

    def check_perplexity(self, sent):
        perplexity = calc_perplexity(self.fluency_model, self.fluency_tokenizer, sent).item()

        # LOGGING
        # print(f'[LOG]: Perplexity: {perplexity}')

        return perplexity < self.fluency_threshold

    def check_relevancy(self, post, response):
        tokenized_pair = self.relevancy_tokenizer(post, response, return_tensors='pt')
        loss = self.relevancy_model(**tokenized_pair).item()

        # LOGGING
        # print(f'[LOG]: Relevancy: {loss}')

        return loss > self.relevancy_threshold

    def __call__(self, curr_sent, prev_sent=None):
        perplexity = self.check_perplexity(curr_sent) if prev_sent is None else self.check_perplexity(f'{prev_sent} {curr_sent}')
        relevancy = self.check_relevancy(prev_sent, curr_sent) if prev_sent is not None else True

        return perplexity and relevancy


if __name__ == '__main__':
    judge = JudgeSystem(r'../saved_models/best_fluency_model',
                        '../saved_models/best_relevancy_model/relevancy_model.pt')

    sent1 = "Bu bir soru mu?"
    ans1 = "Yok, değil aslında."
    print(sent1)
    print(judge(curr_sent=ans1, prev_sent=sent1))
    print()
