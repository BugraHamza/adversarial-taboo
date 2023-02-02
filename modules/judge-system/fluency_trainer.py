import time

import optuna as optuna
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup

from modules.utils.util import get_gpt_tokenizer
from modules.utils.custom_datasets import FluencyDataset
from modules.utils.custom_models import get_fluency_model


def train(model, train_loader, optimizer, scheduler):
    model.train()
    pbar = tqdm(train_loader)
    losses = []

    for batch in pbar:
        optimizer.zero_grad()
        loss = model(**batch, labels=batch['input_ids']).loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        pbar.set_description(f'Loss: {np.mean(losses):.5f}')

    return np.mean(losses)


def evaluate(model, val_loader):
    model.eval()
    pbar = tqdm(val_loader)
    losses = []
    with torch.no_grad():
        for batch in pbar:
            loss = model(**batch, labels=batch['input_ids']).loss
            losses.append(loss.item())
            pbar.set_description(f'Loss: {np.mean(losses):.5f}')

    return np.mean(losses)


def main(data_name, model_name, batch_size, num_epochs, device='cpu'):
    # load data
    if data_name == 'reddit':
        train_data = pd.read_parquet('datasets/reddit-dataset/tr-reddit_train.parquet')
        val_data = pd.read_parquet('datasets/reddit-dataset/tr-reddit_val.parquet')
        # test_data = pd.read_parquet('../../datasets/reddit-dataset/te-reddit.parquet')
    elif data_name == 'forum_dh':
        train_data = pd.read_parquet('datasets/donanim-haber-dataset/forum_dh_train.parquet')
        val_data = pd.read_parquet('datasets/donanim-haber-dataset/forum_dh_val.parquet')
        # test_data = pd.read_parquet('../../datasets/donanim-haber-dataset/forum_dh_test.parquet')
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
    optimizer = optim.AdamW(fluency_model.parameters(), lr=0.01)

    # create a learning rate scheduler
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = 0
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    train(fluency_model, train_loader, optimizer, scheduler)
    val_loss = evaluate(fluency_model, val_loader)

    return val_loss


def objective(trial):
    data_name = 'reddit'
    model_name = 'redrussianarmy/gpt2-turkish-cased'
    batch_size = trial.suggest_int('batch_size', 1, 32)
    num_epochs = 1
    device = 'cpu'

    return main(data_name, model_name, batch_size, num_epochs, device)


if __name__ == '__main__':
    study = optuna.create_study(study_name=f'fluency_study', storage='sqlite:///fluency_study.db',
                                direction="minimize", load_if_exists=True,
                                pruner=optuna.pruners.SuccessiveHalvingPruner())
    study.optimize(objective, n_trials=50, gc_after_trial=True)
