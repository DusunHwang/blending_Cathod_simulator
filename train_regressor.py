import csv
import argparse
import os
from statistics import mean

def load_data(path):
    xs, ys = [], []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row['x']))
            ys.append(float(row['y']))
    return xs, ys

def mse(preds, targets):
    return sum((p - t) ** 2 for p, t in zip(preds, targets)) / len(targets)

def r2_score(preds, targets):
    t_mean = mean(targets)
    ss_tot = sum((t - t_mean) ** 2 for t in targets)
    ss_res = sum((t - p) ** 2 for p, t in zip(preds, targets))
    return 1 - ss_res / ss_tot

def train(xs, ys, xs_val, ys_val, epochs=50, lr=0.001):
    w = 0.0
    b = 0.0
    log_rows = []
    for epoch in range(1, epochs + 1):
        # gradient descent step
        grad_w = 2 / len(xs) * sum((w * x + b - y) * x for x, y in zip(xs, ys))
        grad_b = 2 / len(xs) * sum(w * x + b - y for x, y in zip(xs, ys))
        w -= lr * grad_w
        b -= lr * grad_b

        train_preds = [w * x + b for x in xs]
        val_preds = [w * x + b for x in xs_val]
        train_m = mse(train_preds, ys)
        val_m = mse(val_preds, ys_val)
        log_rows.append({'epoch': epoch, 'train_mse': train_m, 'val_mse': val_m})
    return w, b, log_rows

def save_logs(log_rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_mse', 'val_mse'])
        writer.writeheader()
        writer.writerows(log_rows)

def save_model(params, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for k, v in params.items():
            f.write(f"{k}:{v}\n")

def save_report(metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")


def save_placeholder_plot(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 1x1 black png
    data = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"\
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"\
            b"\x00\x00\x00\x0cIDAT\x08\xd7c```\x00\x00\x00\x04"\
            b"\x00\x01\xef\xcc\xcd\x91\x00\x00\x00\x00IEND\xaeB`\x82")
    with open(path, 'wb') as f:
        f.write(data)


def main(args):
    xs_train, ys_train = load_data('data/train.csv')
    xs_val, ys_val = load_data('data/val.csv')
    xs_test, ys_test = load_data('data/test.csv')

    w, b, logs = train(xs_train, ys_train, xs_val, ys_val, epochs=args.epochs, lr=args.lr)

    save_logs(logs, 'logs/train_log.csv')
    save_model({'w': w, 'b': b}, 'models/final_model.pt')
    val_preds = [w * x + b for x in xs_val]
    test_preds = [w * x + b for x in xs_test]

    val_r2 = r2_score(val_preds, ys_val)
    test_r2 = r2_score(test_preds, ys_test)
    val_mse = mse(val_preds, ys_val)
    test_mse = mse(test_preds, ys_test)

    metrics = {
        'val_r2': val_r2,
        'test_r2': test_r2,
        'val_mse': val_mse,
        'test_mse': test_mse,
    }
    if val_r2 > 0.85:
        metrics['high_performance'] = True
    save_report(metrics, 'reports/eval.md')
    save_placeholder_plot('plots/train_vs_test.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    main(args)
