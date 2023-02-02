import flair
import torch
from modules.utils import get_gpt_tokenizer
import pandas as pd
import re

from flair.data import Sentence
from flair.models import SequenceTagger

from modules.utils import concept_generator

from tqdm import tqdm


class Attacker:
    def __init__(self, method, source=None, question_generation_model=None, pos_tagger=None):
        self.device = 'cpu'
        flair.device = self.device

        self.method = method if method in ['direct_inquiry', 'indirect_inducement', 'user'] else 'direct_inquiry'

        self.source = pd.read_csv(source)if source else None

        self.question_generation_model = question_generation_model if question_generation_model else None
        self.pos_tagger = pos_tagger if pos_tagger else None

        if self.question_generation_model:
            self.model = torch.load(question_generation_model, map_location=self.device)
            self.tokenizer = get_gpt_tokenizer(self.model.config.name_or_path)
        elif self.pos_tagger:
            self.pos_tagger = SequenceTagger.load(pos_tagger)

        self.curr_contents = None
        self.curr_concepts = None
        self.word = None
        self.question_word = None   # this word is used to generate the question

    def assign_word(self, word):
        self.word = word

        if self.method == 'direct_inquiry':
            self.question_word = word

        elif self.method == 'indirect_inducement':
            self.curr_concepts = concept_generator(word)
            self.question_word = next(self.curr_concepts)

        self.curr_contents = self.content_generator()

    def reassign_word(self):
        self.question_word = next(self.curr_concepts)

    def content_generator(self):
        self.source = self.source.sample(frac=1).reset_index(drop=True)

        for content in self.source['content']:
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', content)
            for i, sentence in enumerate(sentences):
                if f' {self.question_word} ' in sentence and len(re.findall(self.question_word, sentence)) == 1:
                    yield sentence
        yield self.question_word

    def find_content(self):
        return self.curr_contents.__next__()

    def ask(self):
        if self.method == 'user':
            user_question = input('Please enter your question: ')
            return user_question
        elif self.question_generation_model:
            return self.ask_generation_based()
        elif self.pos_tagger:
            return self.ask_rule_based()

    def ask_generation_based(self):
        content = self.find_content()
        el_sent = f"{self.tokenizer.bos_token}Paragraf: {content} Soru: "

        input_ids = self.tokenizer(el_sent, return_tensors='pt').input_ids
        generated = self.model.generate(input_ids, do_sample=True, max_length=100,
                                        exponential_decay_length_penalty=[2.5, -3.0],
                                        typical_p=0.6, num_return_sequences=1,
                                        temperature=0.8)

        tokenized_generated = self.tokenizer.decode(generated[0][len(input_ids[0]):], skip_special_tokens=True)
        return tokenized_generated

    def ask_rule_based(self):
        content = self.find_content()

        sentence = Sentence(content)
        self.pos_tagger.predict(sentence)

        question = []
        for token in sentence:
            label = token.get_tag('upos').value
            if token.text == self.question_word:
                if label == 'NOUN':
                    question.append('ne')
                else:
                    return None
            else:
                question.append(token.text)

        if question[-1] in '.!':
            question.pop()
        question.append('?')

        return ' '.join(question)


if __name__ == '__main__':
    attacker = Attacker(pos_tagger='PoS_Tag_Question_Generation/flert-imst-dbmdz/bert-base-turkish-cased-42/final-model.pt',
                        # question_generation_model='/Users/bugrahamzagundog/Desktop/Courses/AutoTaboo-Player/modules/OpenAttacker/GPT_Based_Question_Generation/question_generation_model_6.776570192980997',
                        source='../../datasets/tr_wiki.csv',
                        method='indirect_inducement')
    word = 'sözcük'
    attacker.assign_word(word)

    question = attacker.ask()
    print('Attacker: ', question)
    print('Attacker question word: ', attacker.question_word)
