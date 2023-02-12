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

best_acc = 0


def calc_accuracy(y_pred, y, relevancy_threshold=0.4):
    return ((y_pred > relevancy_threshold) == y).sum().item() / len(y)


def train(model, train_loader, criterion, optimizer, scheduler, relevancy_threshold=0.4):
    model.train()
    pbar = tqdm(train_loader)
    losses, accuracies = [], []

    for batch, y in pbar:
        optimizer.zero_grad()
        y_pred = model(**batch).reshape(len(y))
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        accuracies.append(calc_accuracy(y_pred, y, relevancy_threshold=relevancy_threshold))
        pbar.set_description(f'Loss: {np.mean(losses):.5f} - Accuracy: {np.mean(accuracies):.5f}')

    return np.mean(accuracies)


def evaluate(model, val_loader, criterion, relevancy_threshold=0.4):
    model.eval()
    pbar = tqdm(val_loader)
    losses, accuracies = [], []

    with torch.no_grad():
        for batch, y in pbar:
            y_pred = model(**batch).reshape(len(y))
            loss = criterion(y_pred, y)

            losses.append(loss.item())
            accuracies.append(calc_accuracy(y_pred, y, relevancy_threshold=relevancy_threshold))
            pbar.set_description(f'Loss: {np.mean(losses):.5f} - Accuracy: {np.mean(accuracies):.5f}')

    return np.mean(accuracies)


def train_val_fn(data_name, model_name, batch_size, learning_rate, num_epochs,
                 device='cpu', threshold=0.4, trainable_llm=False):
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
    relevancy_model = get_relevancy_model(model_name, trainable_llm=trainable_llm, device=device)

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

        # if val_acc > best_val_acc:
        #     torch.save(relevancy_model, f'modules/judge-system/best_fluency_model/relevancy_{batch_size}_{learning_rate}_{num_epochs}_{threshold}_{int(1000*val_acc)}.pt')
        #     best_val_acc = val_acc

    return relevancy_model, val_acc


def objective(trial, data_name, device):
    global best_acc

    data_name = data_name
    model_name = 'dbmdz/bert-base-turkish-cased'
    batch_size = trial.suggest_int('batch_size', 1, 32)
    learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-2, log=True)
    num_epochs = 1  # trial.suggest_int('num_epochs', 1, 5)
    threshold = trial.suggest_float('threshold', 0.1, 0.9)

    model, acc = train_val_fn(data_name, model_name, batch_size, learning_rate, num_epochs, device, threshold)

    if acc > best_acc:
        torch.save(model, f'best_model.pt')
        print('='*50)
        print('CURRENT BEST MODEL SAVED!')
        print('='*50)
        print(f'Batch Size: {batch_size} - Learning Rate: {learning_rate} - Num Epochs: {num_epochs} - Threshold: {threshold} - Accuracy: {acc:.5f}')
        best_acc = acc

    return acc


def main(study_name, data_name, num_trials, device):
    study = optuna.create_study(study_name=study_name, storage=f'sqlite:///{study_name}.db',
                                direction="maximize", load_if_exists=True,
                                pruner=optuna.pruners.SuccessiveHalvingPruner())
    study.optimize(lambda x: objective(x, data_name, device), n_trials=num_trials, gc_after_trial=True)


if __name__ == '__main__':
    """
    from multiprocessing import Process

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='reddit')
    parser.add_argument('--study_name', type=str, default='relevancy_study')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_trials', type=int, default=50)
    parser.add_argument('--num-processes', type=int, default=1)

    args = parser.parse_args()

    if args.device == 'cpu' or args.device == 'cuda' or args.device == 'mps':
        main(args.study_name, args.data_name, args.num_trials, args.device)

    elif args.device == 'double_cuda':
        p1 = Process(target=main, args=(args.study_name, args.data_name, args.num_trials, 'cuda:0'))
        p2 = Process(target=main, args=(args.study_name, args.data_name, args.num_trials, 'cuda:1'))

        p1.start()
        p2.start()

        p1.join()
        p2.join()

    else:
        raise ValueError('Invalid device name! (cpu, cuda, double_cuda). You are probably using a TPU.')

    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_name', type=str, default='reddit')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--trainable_llm', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    model, _ = train_val_fn(args.data_name, 'dbmdz/bert-base-turkish-cased', args.batch_size, args.learning_rate,
                            args.num_epochs, device=args.device, threshold=0.4, trainable_llm=args.trainable_llm)

    torch.save(model, 'relevancy_model.pt')
