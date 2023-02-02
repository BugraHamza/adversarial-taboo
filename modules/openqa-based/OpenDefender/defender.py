import random
import time
from collections import defaultdict

import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


class Defender:
    def __init__(self, method=None, model=None, source=None):
        self.device = 'cpu'
        self.sentence = None
        self.method = method if method in ['no_defense', 'intention_detection', 'inducement_prevention', 'user'] else 'no_defense'

        if source:
            self.source = pd.read_csv(source)
            self.corpus = self.get_corpus()
            self.bm25 = BM25Okapi([paragraph.split() for paragraph in self.corpus])

        if model:
            self.model = AutoModelForQuestionAnswering.from_pretrained(model)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model.config.name_or_path)
            self.qa_pipeline = pipeline('question-answering', model=self.model, tokenizer=self.tokenizer)

        self.score_threshold = 0.3

        self.curr_possible_answers = None
        self.intended_word = None
        self.answer_traces = []

    def assign_sentence(self, sentence):
        self.sentence = sentence
        self.curr_possible_answers = self.get_answer_list()

    def get_corpus(self):
        corpus = []
        sampled_contents = self.source['content'].sample(frac=.3)
        for content in sampled_contents:
            paras = content.split('\n\n')
            paras_sample = int(len(paras) * .3)
            for paragraph in random.sample(paras, paras_sample):
                corpus.append(paragraph)
        return corpus

    def get_relevant_paragraphs(self):
        top_paragraphs = self.bm25.get_top_n(self.sentence.split(), self.corpus, n=5)
        return top_paragraphs

    def get_answer_list(self):
        relevant_paras = self.get_relevant_paragraphs()

        answers = []
        for para in relevant_paras:
            try:
                result = self.qa_pipeline(question=self.sentence, context=para, max_length=512, top_k=1)
                answers.append((result['answer'], result['score']))
            except:
                pass

        answers.sort(key=lambda x: x[1], reverse=True)
        return answers

    def answer_randomly(self):
        ans_ls = self.get_answer_list()
        if ans_ls:
            d_answer, _ = random.choice(ans_ls)
            return d_answer
        return ''

    def answer(self):
        if self.method == 'user':
            user_sent = input('Please enter your answer: ')
            return user_sent

        if self.curr_possible_answers:
            if self.method == 'no_defense':
                most_probable_answer = self.curr_possible_answers.pop(0)
                return most_probable_answer[0]
            elif self.method == 'intention_detection':
                most_probable_answer, most_probable_score = self.curr_possible_answers.pop(0)
                if most_probable_score > self.score_threshold:
                    self.intended_word = most_probable_answer
                    print(f'Intended word: {self.intended_word}')
                    return ''
                else:
                    return most_probable_answer

            elif self.method == 'inducement_prevention':
                most_probable_answer, most_probable_score = self.curr_possible_answers.pop(0)
                self.answer_traces.append((most_probable_answer, most_probable_score))

                second_most_answer, second_most_score = self.curr_possible_answers.pop(0)
                while most_probable_answer in second_most_answer or most_probable_score < self.score_threshold:
                    second_most_answer, second_most_score = self.curr_possible_answers.pop(0)
                    if not self.curr_possible_answers:
                        return ''

                return second_most_answer

        return None

    def answer_using_traces(self):
        prob_answers = defaultdict(lambda: 0)
        for ans, score in self.answer_traces:
            for word in ans.split():
                prob_answers[word] += score

        print(f"Answer traces: {dict(prob_answers)}")

        max_word = max(prob_answers, key=prob_answers.get)
        return max_word


if __name__ == '__main__':
    defender = Defender('inducement_prevention', 'savasy/bert-base-turkish-squad',
                        '/Users/bugrahamzagundog/Desktop/Courses/AutoTaboo-Player/datasets/tr_wiki.csv')
    sent1 = "sade bir ne yaparlar ve birlikte bir hayata başlarlar ve sonunda iki oğulları olur ?"
    defender.assign_sentence(sent1)
    answer = defender.answer()

    sent2 = "hera bu ağacı gaia'nın kendisine ne hediyesi olarak verdiği meyve ağacı dallarından yetiştirmiş , hesperidleri de bu ağaçlara bakma görevini vermiştir ?"
    defender.assign_sentence(sent2)
    answer = defender.answer()
    print(answer)
    print(defender.answer_using_traces())
