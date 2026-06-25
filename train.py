import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import load_all_subjects, loso_split, WESADDataset, ALL_SUBJECTS
from models import stress_transformer, lightweight_cnn
from utils import compute_metrics, print_metrics, get_class_weights


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='WESAD', help='Path to WESAD folder')
    parser.add_argument('--model', type=str, default='stress_transformer',
                        choices=['stress_transformer', 'lightweight_cnn'],
                        help='Model to train')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs per fold')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--use-gpu', default=False, action='store_true', help='Use GPU')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory')
    return parser.parse_args()


def train_epoch(loader, model, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(loader, model, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1)[:, 1]
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())
    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs))


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    device = torch.device('cuda:0') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')
    print(f'Model: {args.model}')

    print('Loading all subjects...')
    windows, labels, subject_ids = load_all_subjects(args.data_dir)
    print(f'Total windows: {len(windows)}, stress: {labels.sum()}, non-stress: {(labels==0).sum()}')

    # leave-one-subject-out cross validation
    all_metrics = []
    subjects_present = np.unique(subject_ids)

    for test_subject in subjects_present:
        print(f'\n--- LOSO: test subject S{test_subject} ---')
        X_train, y_train, X_test, y_test = loso_split(windows, labels, subject_ids, test_subject)
        print(f'Train: {len(X_train)} windows | Test: {len(X_test)} windows')

        # class weights to fix imbalance — this is what the original failed to do
        class_weights = get_class_weights(y_train)
        weight_tensor = torch.tensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        train_dataset = WESADDataset(X_train, y_train)
        test_dataset = WESADDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        if args.model == 'stress_transformer':
            model = stress_transformer().to(device)
        else:
            model = lightweight_cnn().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        best_f1 = 0
        best_model_path = f'models/{args.model}_S{test_subject}.pth'

        for epoch in range(args.epochs):
            loss = train_epoch(train_loader, model, optimizer, criterion, device)
            y_true, y_pred, y_prob = eval_epoch(test_loader, model, device)
            metrics = compute_metrics(y_true, y_pred, y_prob)
            scheduler.step()

            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                torch.save(model.state_dict(), best_model_path)

            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch+1}/{args.epochs} | Loss: {loss:.4f}', end=' | ')
                print_metrics(metrics)

        # load best model and evaluate
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        y_true, y_pred, y_prob = eval_epoch(test_loader, model, device)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        print(f'Best result for S{test_subject}:', end=' ')
        print_metrics(metrics)
        all_metrics.append(metrics)

    # average across all subjects
    print('\n========== LOSO Cross-Validation Results ==========')
    for key in ['f1', 'recall', 'precision', 'accuracy', 'auc']:
        vals = [m[key] for m in all_metrics if key in m]
        print(f'Avg {key.upper()}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')

    # save summary
    import pandas as pd
    df = pd.DataFrame(all_metrics, index=[f'S{s}' for s in subjects_present])
    df.to_csv(os.path.join(args.results_dir, f'{args.model}_loso_results.csv'))
    print(f'Results saved to results/{args.model}_loso_results.csv')