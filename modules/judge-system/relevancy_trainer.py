import argparse

import numpy as np
import optuna

import torch
import transformers.utils.logging
from torch import optim, nn
from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup, set_seed

from modules.utils.util import get_bert_tokenizer, get_data
from modules.utils.custom_datasets import RelevancyDataset
from modules.utils.custom_models import get_relevancy_model

set_seed(42)
transformers.utils.logging.set_verbosity_error()


def train(model, train_loader, criterion, optimizer, scheduler, relevancy_threshold=0.4):
    model.train()
    pbar = tqdm(train_loader)
    losses, accuracies = [], []

    for batch, y in pbar:
        optimizer.zero_grad()
        y_pred = model(**batch).squeeze()
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        accuracies.append(((y_pred > relevancy_threshold) == y).sum() / len(y))
        pbar.set_description(f'Loss: {np.mean(losses):.5f} - Accuracy: {np.mean(accuracies):.5f}')

    return np.mean(accuracies)


def evaluate(model, val_loader, criterion, relevancy_threshold=0.4):
    model.eval()
    pbar = tqdm(val_loader)
    losses, accuracies = [], []

    with torch.no_grad():
        for batch, y in pbar:
            y_pred = model(**batch).squeeze()
            loss = criterion(y_pred, y)

            losses.append(loss.item())
            accuracies.append(((y_pred > relevancy_threshold) == y).sum() / len(y))
            pbar.set_description(f'Loss: {np.mean(losses):.5f} - Accuracy: {np.mean(accuracies):.5f}')

    return np.mean(accuracies)


def train_val_fn(data_name, model_name, batch_size, learning_rate, num_epochs, device='cpu', threshold=0.4):
    # load data
    train_data, val_data, test_data = get_data(data_name, 'cls')

    # create a tokenizer
    tokenizer = get_bert_tokenizer(model_name)

    train_set = RelevancyDataset(train_data, tokenizer, device=device)
    val_set = RelevancyDataset(val_data, tokenizer, device=device)
    # test_set = RelevancyDataset(test_data, tokenizer, device=device)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # create a model
    relevancy_model = get_relevancy_model(model_name, device=device)

    # create an optimizer
    optimizer = optim.AdamW(relevancy_model.parameters(), lr=learning_rate)

    # create a criterion
    criterion = nn.BCELoss()

    # create a learning rate scheduler
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = num_training_steps // 10
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    best_val_acc = 0
    for _ in range(num_epochs):
        train_acc = train(relevancy_model, train_loader, criterion, optimizer, scheduler, relevancy_threshold=threshold)
        val_acc = evaluate(relevancy_model, val_loader, criterion, relevancy_threshold=threshold)
        print(f'Train Accuracy: {train_acc:.5f} - Val Accuracy: {val_acc:.5f}')

        if val_acc > best_val_acc:
            relevancy_model.save_pretrained('modules/judge_system/saved_relevancy_models')
            best_val_acc = val_acc

    return val_acc


def objective(trial, data_name, device):
    data_name = data_name
    model_name = 'dbmdz/bert-base-turkish-cased'
    batch_size = trial.suggest_int('batch_size', 1, 16)
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
    num_epochs = 2
    threshold = 0.4

    return train_val_fn(data_name, model_name, batch_size, learning_rate, num_epochs, device, threshold)


def main(data_name, num_trials, device):
    study = optuna.create_study(study_name='fluency_study', storage='sqlite:///fluency_study.db', direction="minimize",
                                load_if_exists=True, pruner=optuna.pruners.SuccessiveHalvingPruner())
    study.optimize(lambda x: objective(x, data_name, device), n_trials=num_trials, gc_after_trial=True)


if __name__ == '__main__':
    from multiprocessing import Process

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
