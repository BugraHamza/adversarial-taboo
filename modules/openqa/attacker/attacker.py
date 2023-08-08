import random
import re

import pandas as pd
import torch
from torch import nn
from zemberek import TurkishSentenceExtractor
from modules.utils.util import concept_generator


def create_text(**kwargs):
    answer = kwargs['answer']
    cloze = kwargs['cloze']
    return f"answer: {answer} context: {cloze}"


class Attacker:
    def __init__(self, method, source='taboo-datasets/turkish-wiki-dataset/tr_wiki.parquet',
                 cloze_model=TurkishSentenceExtractor(), question_generation_model=None,
                 encoder_tokenizer=None, decoder_tokenizer=None):
        self.device = 'cpu'
        self.method = method if method in ['direct_inquiry', 'indirect_inducement', 'user'] else 'direct_inquiry'
        self.wiki = pd.read_parquet(source)
        # self.wiki.content = self._normalize_wiki()

        self.cloze_model = cloze_model
        self.model = question_generation_model
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer if decoder_tokenizer is not None else encoder_tokenizer

    def _normalize_wiki(self):
        return self.wiki.content.str.replace(r'\s', ' ')

    def _get_word(self, word):
        # word = f' {word.strip()} '
        if self.method == 'direct_inquiry':
            return word
        elif self.method == 'indirect_inducement':
            return concept_generator(word)

    def _get_context(self, word):
        contain_check = rf'{word}|{word.capitalize()}|{word.lower()}|{word.upper()}'
        yield self.wiki.content[self.wiki.content.str.contains(contain_check)].sample(n=1).values[0]

    def _get_cloze(self, context, word):
        sentences = self.cloze_model.from_paragraph(context)  # context.replace('\n', ' ')
        clozes = [sentence for sentence in sentences if sentence.count(word) == 1]
        random.shuffle(clozes)

        for cloze in clozes:
            yield cloze

    @torch.no_grad()
    def _generate_question(self, cloze: str, word: str, num_questions: int = 1, num_beams: int = 1):
        word_index = cloze.index(word)
        cloze = f'{cloze[:word_index]} <hl> {word.strip()} <hl> {cloze[word_index + len(word):]}'
        content = f'answer: {word} context: {cloze}'

        encoder_inputs = self.encoder_tokenizer(content, max_length=256, truncation=True,
                                                return_tensors='pt').input_ids
        bad_word_ids = self.decoder_tokenizer([word.lower(), word.capitalize(), word.upper()],
                                              add_special_tokens=False).input_ids

        generated_ids = self.model.generate(encoder_inputs,
                                            num_beams=num_beams, max_length=50, do_sample=True,
                                            top_k=50, top_p=0.95, early_stopping=True,
                                            pad_token_id=self.decoder_tokenizer.eos_token_id,
                                            eos_token_id=self.decoder_tokenizer.eos_token_id,
                                            no_repeat_ngram_size=3,
                                            bad_words_ids=bad_word_ids,
                                            num_return_sequences=num_questions)

        return self.decoder_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def ask(self, word: str, num_questions: int = 3, num_beams: int = 5):
        word = self._get_word(word)
        for context in self._get_context(word):
            # print('[DEBUG]: ', context)
            for cloze in self._get_cloze(context, word):
                # cloze = re.sub(r'\n+', '', cloze)
                if word in cloze:
                    # print('[DEBUG]: ', cloze)2
                    for question in set(self._generate_question(cloze, word, num_questions=num_questions, num_beams=num_beams)):
                        # print('[DEBUG]: ', question)
                        # ensure that the question does not contain the word
                        return question
        return ''  # since no question is generated, judge system will force attacker to retry


if __name__ == '__main__':
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, EncoderDecoderModel

    encoder_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    decoder_tokenizer = AutoTokenizer.from_pretrained("redrussianarmy/gpt2-turkish-cased")
    decoder_tokenizer.add_special_tokens({'bos_token': '<BOS>', 'pad_token': '<PAD>', 'eos_token': '<EOS>'})

    encoder_model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased").to('cpu')
    decoder_model = AutoModelForCausalLM.from_pretrained("redrussianarmy/gpt2-turkish-cased",
                                                         add_cross_attention=True).to('cpu')
    decoder_model.resize_token_embeddings(len(decoder_tokenizer))
    decoder_model.config.add_cross_attention = True

    QuestionGenerationModel = EncoderDecoderModel(encoder=encoder_model, decoder=decoder_model).to('cpu')

    attacker = Attacker(source='taboo-datasets/turkish-wiki-dataset/tr_wiki.parquet', method='direct_inquiry',
                        question_generation_model=QuestionGenerationModel,
                        encoder_tokenizer=encoder_tokenizer, decoder_tokenizer=decoder_tokenizer)

    word = 'masa'
    question = attacker.ask(word, num_questions=3, num_beams=15)

    print('Attacker: ', question)
