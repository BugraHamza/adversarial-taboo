import argparse
from multiprocessing import Process

import optuna as optuna
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup, set_seed

from modules.utils.util import get_gpt_tokenizer
from modules.utils.custom_datasets import FluencyDataset
from modules.utils.custom_models import get_fluency_model


set_seed(42)


def train(model, train_loader, optimizer, scheduler):
    model.train()
    pbar = tqdm(train_loader)
    ppls = []

    for batch in pbar:
        optimizer.zero_grad()
        loss = model(**batch, labels=batch['input_ids']).loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        ppls.append(np.exp(loss.item()))
        pbar.set_description(f'PPL: {np.mean(ppls):.5f}')

    return np.mean(ppls)


def evaluate(model, val_loader):
    model.eval()
    pbar = tqdm(val_loader)
    ppls = []
    with torch.no_grad():
        for batch in pbar:
            loss = model(**batch, labels=batch['input_ids']).loss
            ppls.append(np.exp(loss.item()))
            pbar.set_description(f'PPL: {np.mean(ppls):.5f}')

    return np.mean(ppls)


def train_val_fn(data_name, model_name, batch_size, learning_rate, num_epochs, device='cpu'):
    # load data
    if data_name == 'reddit':
        train_data = pd.read_parquet('datasets/reddit-dataset/tr-reddit_train.parquet')
        val_data = pd.read_parquet('datasets/reddit-dataset/tr-reddit_val.parquet')
        # test_data = pd.read_parquet('datasets/reddit-dataset/te-reddit.parquet')
    elif data_name == 'forum_dh':
        train_data = pd.read_parquet('datasets/donanim-haber-dataset/forum_dh_train.parquet')
        val_data = pd.read_parquet('datasets/donanim-haber-dataset/forum_dh_val.parquet')
        # test_data = pd.read_parquet('datasets/donanim-haber-dataset/forum_dh_test.parquet')
    else:
        raise ValueError('Invalid data name!')

    # create a tokenizer
    tokenizer = get_gpt_tokenizer(model_name, max_len=256)

    train_set = FluencyDataset(train_data, tokenizer, device=device)
    val_set = FluencyDataset(val_data, tokenizer, device=device)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # create a model
    fluency_model = get_fluency_model(model_name, tokenizer_length=len(tokenizer), device=device)

    # create an optimizer
    optimizer = optim.AdamW(fluency_model.parameters(), lr=learning_rate)

    # create a learning rate scheduler
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = num_training_steps // 10
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    best_val_ppl = np.inf
    for _ in range(num_epochs):
        train_ppl = train(fluency_model, train_loader, optimizer, scheduler)
        val_ppl = evaluate(fluency_model, val_loader)
        print(f'Train PPL: {train_ppl:.5f} - Val PPL: {val_ppl:.5f}')

        if val_ppl < best_val_ppl:
            fluency_model.save_pretrained('modules/judge_system/saved_fluency_models')
            best_val_ppl = val_ppl

    return val_ppl


def objective(trial, data_name, device):
    data_name = data_name
    model_name = 'redrussianarmy/gpt2-turkish-cased'
    batch_size = trial.suggest_int('batch_size', 1, 16)
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
    num_epochs = 1

    return train_val_fn(data_name, model_name, batch_size, learning_rate, num_epochs, device)


def main(data_name, num_trials, device):
    study = optuna.create_study(study_name='fluency_study', storage='sqlite:///fluency_study.db', direction="minimize",
                                load_if_exists=True, pruner=optuna.pruners.SuccessiveHalvingPruner())
    study.optimize(lambda x: objective(x, data_name, device), n_trials=num_trials, gc_after_trial=True)


if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='reddit')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_trials', type=int, default=50)

    args = parser.parse_args()

    if args.device == 'cpu' or args.device == 'cuda':
        main(args.data_name, args.num_trials, args.device)

    elif args.device == 'double_cuda':
        p1 = Process(target=main, args=(args.data_name, args.num_trials, 'cuda:0'))
        p2 = Process(target=main, args=(args.data_name, args.num_trials, 'cuda:1'))

        p1.start()
        p2.start()

        p1.join()
        p2.join()

    else:
        raise ValueError('Invalid device name! (cpu, cuda, double_cuda). You are probably using a TPU.')
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='reddit')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--epochs', type=int, default=50)

    args = parser.parse_args()

    train_val_fn(args.data_name, model_name='redrussianarmy/gpt2-turkish-cased',
                 batch_size=16, learning_rate=0.0002,
                 num_epochs=args.epochs, device=args.device)
