import random
import re

import pandas as pd
from zemberek import TurkishSentenceExtractor
from modules.utils.util import concept_generator


class Attacker:
    def __init__(self, method, source='taboo-datasets/turkish-wiki-dataset/tr_wiki.parquet',
                 cloze_model=TurkishSentenceExtractor(), question_generation_model=None,
                 question_generation_tokenizer=None):
        self.device = 'cpu'
        self.method = method if method in ['direct_inquiry', 'indirect_inducement', 'user'] else 'direct_inquiry'
        self.wiki = pd.read_parquet(source)
        self.wiki.content = self._normalize_wiki()

        self.cloze_model = cloze_model
        self.model = question_generation_model
        self.tokenizer = question_generation_tokenizer

    def _normalize_wiki(self):
        return self.wiki.content.str.replace(r'\s', ' ')

    def _get_word(self, word):
        # word = f' {word.strip()} '
        if self.method == 'direct_inquiry':
            return word
        elif self.method == 'indirect_inducement':
            return concept_generator(word)

    def _get_context(self, word):
        word = f' {word.strip()} '
        yield self.wiki.content[self.wiki.content.str.contains(rf'{word}|{word.capitalize()}')].sample(n=1).values[0]

    def _get_cloze(self, context, word):
        sentences = self.cloze_model.from_paragraph(context.replace('\n', ' '))
        clozes = [sentence for sentence in sentences if sentence.count(word) >= 1]
        cloze = random.choice(clozes)
        return cloze

    def _generate_question(self, cloze: str, word: str, num_questions: int = 3, num_beams: int = 5):
        bad_words_ids = self.tokenizer([word, word.capitalize()], add_special_tokens=False).input_ids

        text = f'answer: {word} context: {cloze}'
        tokenized_sent = self.tokenizer(text, padding='max_length', max_length=256,
                                        truncation=True, return_tensors='pt').to(self.device)
        generated_question = self.model.generate(tokenized_sent['input_ids'], max_length=128, do_sample=True, top_k=50,
                                                 top_p=0.95, num_beams=num_beams, num_return_sequences=num_questions,
                                                 no_repeat_ngram_size=2, early_stopping=True,
                                                 bad_words_ids=bad_words_ids)

        return self.tokenizer.batch_decode(generated_question, skip_special_tokens=True)

    def ask(self, word: str):
        word = self._get_word(word)
        for context in self._get_context(word):
            for cloze in self._get_cloze(context, word):
                for question in self._generate_question(cloze, word):
                    # ensure that the question does not contain the word
                    if word not in question:
                        return question
                return ''  # since no question is generated, judge system will force attacker to retry


if __name__ == '__main__':
    from transformers import MT5ForConditionalGeneration, MT5TokenizerFast, AutoConfig
    model_path = 'modules/openqa/attacker/mt5-base-3task-highlight-combined3'
    config = AutoConfig.from_pretrained(model_path)
    QuestionGenerationModel = MT5ForConditionalGeneration.from_pretrained(model_path, config=config)
    QuestionGenerationTokenizer = MT5TokenizerFast.from_pretrained(model_path, config=config)

    attacker = Attacker(source='taboo-datasets/turkish-wiki-dataset/tr_wiki.parquet', method='direct_inquiry',
                        question_generation_model=QuestionGenerationModel,
                        question_generation_tokenizer=QuestionGenerationTokenizer)

    word = 'masa'
    question = attacker.ask(word)

    print('Attacker: ', question)
