import argparse
import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


def load_dataset(path):
    df = pd.read_csv(path)
    x = torch.tensor(df['x'].values, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(df['y'].values, dtype=torch.float32).unsqueeze(1)
    return TensorDataset(x, y)


def train(model, train_ds, val_ds, epochs=50, lr=0.01, device='cpu'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log_rows = []
    train_loader = DataLoader(train_ds, batch_size=len(train_ds))
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_preds = model(train_ds.tensors[0].to(device))
            val_preds = model(val_ds.tensors[0].to(device))
            train_loss = criterion(train_preds, train_ds.tensors[1].to(device)).item()
            val_loss = criterion(val_preds, val_ds.tensors[1].to(device)).item()
        log_rows.append({'epoch': epoch, 'train_mse': train_loss, 'val_mse': val_loss})
    return log_rows


def evaluate(model, dataset, device='cpu'):
    model.eval()
    x, y = dataset.tensors
    with torch.no_grad():
        preds = model(x.to(device)).cpu().numpy().ravel()
    y_true = y.numpy().ravel()
    r2 = r2_score(y_true, preds)
    mse = mean_squared_error(y_true, preds)
    return preds, r2, mse


def save_logs(log_rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(log_rows).to_csv(path, index=False)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def save_report(metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")


def plot_predictions(y_true, preds, r2, mse, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure()
    plt.scatter(y_true, preds, label='predicted')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='ideal')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'R2={r2:.3f}, MSE={mse:.3f}')
    plt.legend()
    plt.savefig(path)
    plt.close()


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_ds = load_dataset('data/train.csv')
    val_ds = load_dataset('data/val.csv')
    test_ds = load_dataset('data/test.csv')

    model = nn.Linear(1, 1)
    logs = train(model, train_ds, val_ds, epochs=args.epochs, lr=args.lr, device=device)

    save_logs(logs, 'logs/train_log.csv')
    save_model(model, 'models/final_model.pt')

    _, val_r2, val_mse = evaluate(model, val_ds, device)
    test_preds, test_r2, test_mse = evaluate(model, test_ds, device)

    metrics = {
        'val_r2': val_r2,
        'val_mse': val_mse,
        'test_r2': test_r2,
        'test_mse': test_mse,
    }
    if val_r2 > 0.85:
        metrics['high_performance'] = True

    save_report(metrics, 'reports/eval.md')

    y_test = test_ds.tensors[1].numpy().ravel()
    plot_predictions(y_test, test_preds, test_r2, test_mse, 'plots/train_vs_test.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    main(args)
