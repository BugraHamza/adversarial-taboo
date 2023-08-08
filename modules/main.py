from __future__ import annotations

import os
import random
import numpy as np
import torch

from transformers import AutoConfig, MT5ForConditionalGeneration, AutoTokenizer, EncoderDecoderModel

from judge_system.judge_system import JudgeSystem
from openqa.attacker.attacker import Attacker
from openqa.defender.defender import Defender

from datetime import datetime

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class Assigner:
    def __init__(self, **kwargs):
        if vocab_file := kwargs.get('vocab_file'):
            with open(vocab_file) as f:
                self.vocab_list = [line[:-1] for line in f.readlines()]
                random.shuffle(self.vocab_list)

            self.vocab_iter = iter(self.vocab_list)

        elif word := kwargs.get('word'):
            self.vocab_list = [word]
            self.vocab_iter = iter(self.vocab_list)

        else:
            raise ValueError('Either vocab_file or word must be given.')

    def __call__(self):
        return next(self.vocab_iter)


class Game:
    def __init__(self, judge_system: JudgeSystem, attacker: Attacker, defender: Defender,
                 n_rounds: int = 5, n_turns: int = 10, vocab_file=None, word=None):
        if judge_system is None or attacker is None or defender is None:
            raise ValueError("Agents must be initialized..")

        # initialize the players and the judge system
        self.judge_system = judge_system
        self.attacker = attacker
        self.defender = defender

        # initialize the game parameters
        self.n_rounds = n_rounds
        self.n_turns = n_turns
        self.results = {'attacker': 0, 'defender': 0, 'draw': 0}

        # initialize the word assigner
        self.assigner = Assigner(vocab_file=vocab_file, word=word)

    def attacker_turn(self, word):
        # get a question from the attacker
        while True:  # TODO: try for n times
            attacker_question = self.attacker.ask(word=word)
            # print('[DEBUG] Attacker question: ', attacker_question)
            if self.judge_system(curr_sent=attacker_question):
                return attacker_question

    def defender_turn(self, question):
        most_related_answer = None
        # get an answer from the defender
        for answer in self.defender.answer(question, N=15):
            if most_related_answer is None:
                most_related_answer = answer
            # print('[DEBUG] Defender answer: ', answer)
            if self.judge_system(curr_sent=answer, prev_sent=question):
                return answer

        return ''  # no answer found, the judge system will force the defender to answer again

    def play_round(self):
        word = f' {self.assigner().lower()} '
        print(f"Assigned word: {word}")

        # play game for n_turns or until check_if_end() returns True
        for turn in range(self.n_turns):
            # attacker turn
            question = self.attacker_turn(word)
            print(f'[{datetime.now()}] Attacker: {question}')

            # defender turn
            answer = self.defender_turn(question)
            print(f'[{datetime.now()}] Defender: {answer}')

            # check if the game is over
            if word in answer.lower():
                return 'attacker'

        else:
            # if the last turn is reached
            # give the defender a last chance to guess the word
            # self.defender_turn()
            prediction = self.defender_turn(question)
            if word in prediction.lower():
                return 'defender'
            else:
                return 'draw'

    def play_game(self, verbose=False):
        if verbose:
            print(f"Playing game with {self.attacker.method} and {self.defender.method}")

        # play game for n_rounds
        for round in range(self.n_rounds):
            print(f"===== Round {round + 1} =====")
            round_results = self.play_round()

            if round_results == 'attacker':
                self.results['attacker'] += 1
            elif round_results == 'defender':
                self.results['defender'] += 1
            else:
                self.results['draw'] += 1

            if verbose:
                self.print_results()

    def print_results(self):
        print("==============================")
        print("           Results            ")
        print("==============================")
        print(f"Attacker won {self.results['attacker']} times")
        print(f"Defender won {self.results['defender']} times")
        print(f"Draw {self.results['draw']} times")
        print("==============================")


if __name__ == '__main__':
    judge = JudgeSystem(fluency_path='modules/saved_models/best_fluency_model',
                        relevancy_path='modules/saved_models/best_relevancy_model/best_model.pt',
                        fluency_threshold=250, relevancy_threshold=0.4)  # try 0.3752115408021162
    print(f'Judge system initialized with fluency threshold: {judge.fluency_threshold} and relevancy threshold: {judge.relevancy_threshold}')
    model_path = 'obss/mt5-small-3task-both-tquad2'
    config = AutoConfig.from_pretrained(model_path)
    attacker_model = MT5ForConditionalGeneration.from_pretrained(model_path, config=config)
    attacker_tokenizer = AutoTokenizer.from_pretrained(model_path, config=config, max_length=256)

    # model_path = 'modules/saved_models/attacker-models/bert-gpt2'
    # attacker_model = EncoderDecoderModel.from_pretrained(model_path)
    # encoder_tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, 'encoder'), max_length=256)
    # decoder_tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, 'decoder'),  max_length=256)

    attacker = Attacker(source='taboo-datasets/turkish-wiki-dataset/tr_wiki.parquet', method='direct_inquiry',
                        question_generation_model=attacker_model,
                        encoder_tokenizer=attacker_tokenizer)

    # ['no_defense', 'intention_detection', 'inducement_prevention', 'user']
    defender = Defender(source='taboo-datasets/turkish-wiki-dataset/tr_wiki.parquet',
                        model='husnu/bert-base-turkish-cased-finetuned_lr-2e-05_epochs-3',
                        method='no_defense')

    game = Game(vocab_file='word-selection/trial_words.txt',
                n_rounds=30, n_turns=10,
                judge_system=judge, attacker=attacker, defender=defender)

    game.play_game(verbose=True)

    game.print_results()
