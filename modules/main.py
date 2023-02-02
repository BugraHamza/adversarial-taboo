import sys

from JudgeSystem.judge_system import JudgeSystem
from OpenAttacker.attacker import Attacker
from OpenDefender.defender import Defender

from datetime import datetime

from warnings import filterwarnings

filterwarnings('ignore')

terminal = sys.stdout
sys.stdout = open('logs-strategy5.txt', 'w')


class Assigner:
    def __init__(self, vocab_file=None, word=None):
        if vocab_file:
            with open(vocab_file) as f:
                self.vocab_list = [line[:-1] for line in f.readlines()]
                # random.shuffle(self.vocab_list)

            self.vocab_iter = iter(self.vocab_list)

        elif word:
            self.vocab_list = [word]
            self.vocab_iter = iter(self.vocab_list)

        else:
            raise ValueError('Either vocab_file or word must be specified')

    def __call__(self):
        return next(self.vocab_iter)


class Game:
    def __init__(self, n_rounds=5, n_turns=10, vocab_file=None, word=None, judge_system=None,
                 attacker=None, defender=None):
        if judge_system is None or attacker is None or defender is None:
            raise ValueError("Agents must be initialized..")

        self.n_rounds = n_rounds
        self.n_turns = n_turns
        self.assigner = Assigner(vocab_file=vocab_file, word=word)
        self.judge_system = judge_system
        self.attacker = attacker
        self.defender = defender

        self.attacker_sent = None
        self.defender_sent = None

        self.defender_win = 0
        self.attacker_win = 0
        self.tie = 0

    def assign_word(self):
        # get a word from vocabulary list
        # assign the word to attacker
        # return the word to print it out
        word = self.assigner().lower()
        return word

    def attacker_turn(self):
        # get a question from the attacker
        a_question = None

        while a_question is None or \
                not self.judge_system(curr_sent=a_question, prev_sent=self.defender_sent):
            try:
                a_question = self.attacker.ask()
            except StopIteration:
                self.attacker.reassign_word()

            if a_question == self.attacker.question_word:
                break

        self.attacker_sent = a_question.lower()

    def defender_turn(self):
        # get an answer from the defender
        d_answer = None
        self.defender.assign_sentence(self.attacker_sent)

        while d_answer is None or \
                not self.judge_system(curr_sent=d_answer, prev_sent=self.attacker_sent):
            d_answer = self.defender.answer()

            if d_answer is None:
                break

        if d_answer is None:
            d_answer = self.defender.answer_randomly()

        self.defender_sent = d_answer.lower()

    def check_if_end(self):
        # to avoid ending the game at the first turn
        if self.defender_sent is None:
            return False

        # decide whether the game is ended or not based on the defense strategy
        if self.attacker.word in self.defender_sent:
            return True

        if self.defender.method == 'intention_detection':
            if self.defender.intended_word and self.attacker.word in self.defender.intended_word:
                return True

    def play_turn(self):
        # attacker turn
        self.attacker_turn()
        print(f'[{datetime.now()}] Attacker: {self.attacker_sent}')

        # defender turn
        self.defender_turn()
        print(f'[{datetime.now()}] Defender: {self.defender_sent}')

    def play_round(self):
        word = self.assign_word()
        self.attacker.assign_word(word)
        print(f"Assigned word: {word}")

        # play game for n_turns or until check_if_end() returns True
        turn = 0
        while turn < self.n_turns and not self.check_if_end():
            self.play_turn()
            turn += 1
            terminal.write(f"\tTurn: {turn}\n")

        if turn == self.n_turns:
            # if the game is ended at the last turn
            # give the defender a last chance to guess the word
            # self.defender_turn()

            if self.defender.method == 'inducement_prevention':
                d_answer = self.defender.answer_using_traces()
            else:
                d_answer = self.defender.answer_randomly()

            self.defender_sent = d_answer.lower()

            print(f"Defender's last guess: {self.defender_sent}")

            # if the word is guessed correctly
            # the defender wins
            # otherwise, a tie
            if self.defender.sentence == self.attacker.word:
                self.defender_win += 1
                print("Defender won!")
            else:
                self.tie += 1
                print('It is a tie!')
        else:
            # if the game is ended at any other turn
            # the target word is said unintentionally by the defender
            # so, the attacker wins
            if self.defender.method == 'no_defense':
                self.attacker_win += 1
                print("Attacker won!")

            elif self.defender.method == 'intention_detection' and self.defender.intended_word:
                if self.attacker.word in self.defender.intended_word.lower():
                    self.defender_win += 1
                    print("Defender won!")
                elif self.attacker.word in self.defender_sent.lower():
                    self.attacker_win += 1
                    print("Attacker won!")

            else:
                self.attacker_win += 1
                print("Attacker won!")

    def play_game(self, verbose=False):
        if verbose:
            print(f"Playing game with {self.attacker.method} and {self.defender.method}")

        # play game for n_rounds
        round = 1
        while round <= self.n_rounds:
            try:
                terminal.write(f"Round: {round}\n")
                self.play_round()
                round += 1
            except:
                print(f"Game ended prematurely for {self.attacker.word}\n")
                pass
            print("\n")

            if verbose:
                self.print_results()

    def print_results(self):
        print("==============================")
        print("           Results            ")
        print("==============================")
        print(f"Attacker won {self.attacker_win} times")
        print(f"Defender won {self.defender_win} times")
        print(f"Tie {self.tie} times")
        print("==============================")


if __name__ == '__main__':
    judge = JudgeSystem(fluency_path="judge-system/fluency_model_19.7776380211726",
                        relevancy_path="judge-system/adequacy_model_0.5374999642372131",
                        fluency_threshold=60e3, relevancy_threshold=0.4)

    # ['direct_inquiry', 'indirect_inducement', 'user']
    attacker_agent = Attacker(method='indirect_inducement',
                              pos_tagger='OpenAttacker/PoS_Tag_Question_Generation/flert-imst-dbmdz/bert-base-turkish-cased-42/final-model.pt',
                              source='../datasets/tr_wiki.csv'
                              )

    # ['no_defense', 'intention_detection', 'inducement_prevention', 'user']
    defender_agent = Defender(method='intention_detection',
                              model='savasy/bert-base-turkish-squad',
                              source='../datasets/tr_wiki.csv',
                              )

    game = Game( vocab_file='Word Selection/selected_words_mini.txt',
                n_rounds=30, n_turns=10,
                judge_system=judge, attacker=attacker_agent, defender=defender_agent)

    game.play_game(verbose=False)

    game.print_results()
