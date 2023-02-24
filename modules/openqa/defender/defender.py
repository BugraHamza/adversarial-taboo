import random
import time
from collections import defaultdict

import pandas as pd
from rank_bm25 import BM25Okapi

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


class BaseDefender:
    def __init__(self, source, model, method='no_defense'):
        self.source = pd.read_parquet(source)
        self.corpus = self.get_corpus()
        self.bm25 = BM25Okapi([paragraph.split() for paragraph in self.corpus])

        self.model = AutoModelForQuestionAnswering.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.config.name_or_path)
        self.qa_pipeline = pipeline('question-answering', model=self.model, tokenizer=self.tokenizer)

        self.score_threshold = 0.3

    def get_corpus(self):
        corpus = []
        sampled_contents = self.source['content'].sample(frac=.3)
        for article in sampled_contents:
            paragraphs = article.split('\n\n')
            paras_sample = int(len(paragraphs) * .3)
            for paragraph in random.sample(paragraphs, paras_sample):
                corpus.append(paragraph)
        return corpus

    def get_relevant_paragraphs(self, sentence: str, N: int):
        top_paragraphs = self.bm25.get_top_n(sentence.split(), self.corpus, n=N)
        return top_paragraphs

    def get_answer_list(self, sentence: str, N: int = 5):
        relevant_paragraphs = self.get_relevant_paragraphs(sentence=sentence, N=N)

        answers = []
        for paragraph in relevant_paragraphs:
            result = self.qa_pipeline(question=sentence, context=paragraph, max_length=512, top_k=1)
            answers.append((result['answer'], result['score']))

        answers.sort(key=lambda x: x[1], reverse=True)
        return answers

    def answer_randomly(self):
        answers = self.get_answer_list()
        if answers:
            d_answer, _ = random.choice(answers)
            return d_answer
        return ''

    def get_traced_answers(self, sentence: str, N: int):
        prob_answers = defaultdict(lambda: 0)
        for ans, score in self.get_answer_list(sentence=sentence, N=N):
            for word in ans.split():
                prob_answers[word] += score

        print(f"Answer traces: {dict(prob_answers)}")

        max_word = max(prob_answers, key=prob_answers.get)
        return max_word


class NoDefenseDefender(BaseDefender):
    def __init__(self, source, model):
        super().__init__(source=source, model=model, method='no_defense')

    def answer(self, sentence: str, N: int = 5):
        return self.get_answer_list(sentence=sentence, N=N)[0][0]


class IntentionDetectionDefender(BaseDefender):
    def __init__(self, source, model):
        super().__init__(source=source, model=model, method='intention_detection')

    def answer(self, sentence: str, N: int = 5):
        most_probable_answer, most_probable_score = self.get_answer_list(sentence=sentence, N=N)[0]
        answer = '' if most_probable_score < self.score_threshold else most_probable_answer
        return answer


class InducementPreventionDefender(BaseDefender):
    def __init__(self, source, model):
        super().__init__(source=source, model=model, method='inducement_prevention')
        self.traced_answers = []

    def answer(self, sentence: str, N: int = 5):
        answer_ls = self.get_answer_list(sentence=sentence, N=N)

        # get the most probable two answers and their scores
        most_probable_answer, most_probable_score = answer_ls.pop(0)
        second_most_answer, second_most_score = answer_ls.pop(0)

        # append the most probable answer to the traced answers
        self.traced_answers.append(most_probable_answer)

        # find a second most probable answer that does not contain the most probable answer in it
        while most_probable_answer in second_most_answer or most_probable_score < self.score_threshold:
            second_most_answer, second_most_score = answer_ls.pop(0)
            if not answer_ls:
                return ''

        return second_most_answer


class Defender:
    def __init__(self, source, model, method='no_defense'):
        self.method = method
        self.defender = None if method == 'user' else self.get_defender(source=source, model=model)

    def get_defender(self, source, model):
        defender_map = {'no_defense': NoDefenseDefender,
                        'intention_detection': IntentionDetectionDefender,
                        'inducement_prevention': InducementPreventionDefender}

        if self.method in defender_map:
            defender_class = defender_map[self.method]
            return defender_class(source=source, model=model)
        elif self.method == 'user':
            return None
        else:
            raise ValueError('Invalid defense method')

    def answer(self, sentence: str, N: int = 5):
        if self.method == 'user':
            return input('Please enter your answer: ')
        else:
            return self.defender.answer(sentence=sentence, N=N)


if __name__ == '__main__':
    defender = Defender(source='taboo-datasets/turkish-wiki-dataset/tr_wiki.parquet',
                        model='savasy/bert-base-turkish-squad',
                        method='user')
    sent1 = "sade bir ne yaparlar ve birlikte bir hayata başlarlar ve sonunda iki oğulları olur ?"
    ans1 = defender.answer(sent1)
    print('ANSWER1: ', ans1)

    sent2 = """hera bu ağacı gaia'nın kendisine ne hediyesi olarak verdiği meyve ağacı dallarından yetiştirmiş , 
    hesperidleri de bu ağaçlara bakma görevini vermiştir ?"""
    ans2 = defender.answer(sent2)
    print('ANSWER2: ', ans2)
