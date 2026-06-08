#!/usr/bin/env python3
"""Optimize a NumPy neural network with GPyOpt."""

import json
import os

import GPyOpt
import matplotlib.pyplot as plt
import numpy as np


CHECKPOINT_DIR = "bayes_opt_checkpoints"
REPORT_FILE = "bayes_opt.txt"
PLOT_FILE = "bayes_opt_convergence.png"
RUN_HISTORY = []


def make_dataset(seed=0):
    """Create a small non-linear binary classification dataset."""
    rng = np.random.default_rng(seed)
    n = 600
    half = n // 2

    angles_a = rng.uniform(0, np.pi, half)
    angles_b = rng.uniform(0, np.pi, half)

    class_a = np.c_[np.cos(angles_a), np.sin(angles_a)]
    class_b = np.c_[1 - np.cos(angles_b), 1 - np.sin(angles_b) - 0.5]

    X = np.vstack((class_a, class_b))
    X += rng.normal(0, 0.12, X.shape)
    Y = np.vstack((np.zeros((half, 1)), np.ones((half, 1))))

    order = rng.permutation(n)
    X = X[order]
    Y = Y[order]

    split = int(0.8 * n)
    return X[:split], Y[:split], X[split:], Y[split:]


def sigmoid(Z):
    """Calculate the sigmoid activation."""
    return 1 / (1 + np.exp(-Z))


def accuracy(Y_true, Y_pred):
    """Calculate binary classification accuracy."""
    labels = (Y_pred >= 0.5).astype(int)
    return np.mean(labels == Y_true)


def unpack_params(row):
    """Convert one GPyOpt row into model hyperparameters."""
    log_lr, units, dropout, log_l2, batch_size = row

    return {
        "learning_rate": 10 ** float(log_lr),
        "hidden_units": int(units),
        "dropout_rate": float(dropout),
        "l2": 10 ** float(log_l2),
        "batch_size": int(batch_size),
    }


def checkpoint_name(params, run_id):
    """Build a checkpoint filename containing the hyperparameters."""
    return (
        f"run{run_id:03d}_"
        f"lr{params['learning_rate']:.5f}_"
        f"units{params['hidden_units']}_"
        f"drop{params['dropout_rate']:.2f}_"
        f"l2{params['l2']:.6f}_"
        f"batch{params['batch_size']}.npz"
    )


def init_weights(input_dim, hidden_units, rng):
    """Initialize the weights of a one hidden layer neural network."""
    W1 = rng.normal(0, np.sqrt(2 / input_dim), (input_dim, hidden_units))
    b1 = np.zeros((1, hidden_units))
    W2 = rng.normal(0, np.sqrt(2 / hidden_units), (hidden_units, 1))
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2


def forward(X, W1, b1, W2, b2, dropout_rate=0, rng=None, training=False):
    """Run forward propagation through the network."""
    Z1 = np.matmul(X, W1) + b1
    A1 = np.maximum(0, Z1)
    mask = None

    if training and dropout_rate > 0:
        keep_prob = 1 - dropout_rate
        mask = (rng.random(A1.shape) < keep_prob).astype(float)
        A1 = A1 * mask / keep_prob

    Z2 = np.matmul(A1, W2) + b2
    A2 = sigmoid(Z2)
    cache = (Z1, A1, mask)

    return A2, cache


def train_model(params, run_id):
    """Train one model and return the best validation accuracy."""
    X_train, Y_train, X_val, Y_val = make_dataset()
    rng = np.random.default_rng(1000 + run_id)

    W1, b1, W2, b2 = init_weights(X_train.shape[1],
                                  params["hidden_units"], rng)
    best_acc = 0
    best_epoch = 0
    patience = 10
    max_epochs = 80
    checkpoint = os.path.join(CHECKPOINT_DIR,
                              checkpoint_name(params, run_id))

    for epoch in range(max_epochs):
        order = rng.permutation(X_train.shape[0])
        X_epoch = X_train[order]
        Y_epoch = Y_train[order]

        for start in range(0, X_train.shape[0], params["batch_size"]):
            end = start + params["batch_size"]
            X_batch = X_epoch[start:end]
            Y_batch = Y_epoch[start:end]
            m = X_batch.shape[0]

            A2, cache = forward(X_batch, W1, b1, W2, b2,
                                params["dropout_rate"], rng, True)
            Z1, A1, mask = cache

            dZ2 = A2 - Y_batch
            dW2 = np.matmul(A1.T, dZ2) / m + params["l2"] * W2
            db2 = np.sum(dZ2, axis=0, keepdims=True) / m
            dA1 = np.matmul(dZ2, W2.T)

            if mask is not None:
                dA1 = dA1 * mask / (1 - params["dropout_rate"])

            dZ1 = dA1 * (Z1 > 0)
            dW1 = np.matmul(X_batch.T, dZ1) / m + params["l2"] * W1
            db1 = np.sum(dZ1, axis=0, keepdims=True) / m

            W1 -= params["learning_rate"] * dW1
            b1 -= params["learning_rate"] * db1
            W2 -= params["learning_rate"] * dW2
            b2 -= params["learning_rate"] * db2

        val_pred, _ = forward(X_val, W1, b1, W2, b2)
        val_acc = accuracy(Y_val, val_pred)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            np.savez(checkpoint, W1=W1, b1=b1, W2=W2, b2=b2,
                     val_accuracy=best_acc, epoch=epoch,
                     hyperparameters=json.dumps(params))

        if epoch - best_epoch >= patience:
            break

    return best_acc, best_epoch, checkpoint


def objective(x):
    """Objective function minimized by GPyOpt."""
    scores = []

    for row in x:
        params = unpack_params(row)
        run_id = len(RUN_HISTORY) + 1
        best_acc, best_epoch, checkpoint = train_model(params, run_id)
        score = 1 - best_acc

        RUN_HISTORY.append({
            "run": run_id,
            "score": float(score),
            "val_accuracy": float(best_acc),
            "best_epoch": int(best_epoch),
            "checkpoint": checkpoint,
            "hyperparameters": params,
        })
        scores.append(score)

    return np.array(scores).reshape(-1, 1)


def plot_convergence():
    """Save a convergence plot for the optimization."""
    scores = np.array([item["score"] for item in RUN_HISTORY])
    best_scores = np.minimum.accumulate(scores)

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(scores) + 1), 1 - best_scores, marker="o")
    plt.xlabel("Training session")
    plt.ylabel("Best validation accuracy")
    plt.title("Bayesian Optimization Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    plt.close()


def save_report(optimizer):
    """Save a text report of the Bayesian optimization run."""
    best = min(RUN_HISTORY, key=lambda item: item["score"])

    with open(REPORT_FILE, "w", encoding="utf-8") as report:
        report.write("Bayesian Optimization Report\n")
        report.write("============================\n\n")
        report.write("Model: one hidden layer NumPy neural network\n")
        report.write("Dataset: synthetic two-class moon dataset\n")
        report.write("Metric: validation accuracy\n")
        report.write("Objective minimized: 1 - validation accuracy\n")
        report.write("Maximum Bayesian optimization iterations: 30\n\n")

        report.write("Best run\n")
        report.write("--------\n")
        report.write(f"Run: {best['run']}\n")
        report.write(f"Validation accuracy: {best['val_accuracy']:.6f}\n")
        report.write(f"Objective score: {best['score']:.6f}\n")
        report.write(f"Best epoch: {best['best_epoch']}\n")
        report.write(f"Checkpoint: {best['checkpoint']}\n")
        report.write("Hyperparameters:\n")
        for key, value in best["hyperparameters"].items():
            report.write(f"- {key}: {value}\n")

        report.write("\nAll runs\n")
        report.write("--------\n")
        for item in RUN_HISTORY:
            report.write(json.dumps(item, indent=2))
            report.write("\n")

        report.write("\nRaw optimizer x_opt:\n")
        report.write(str(optimizer.x_opt))
        report.write("\nRaw optimizer fx_opt:\n")
        report.write(str(optimizer.fx_opt))
        report.write("\n")


def main():
    """Run Bayesian optimization."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    np.random.seed(0)

    domain = [
        {"name": "log_learning_rate", "type": "continuous",
         "domain": (-4, -1.3)},
        {"name": "hidden_units", "type": "discrete",
         "domain": (8, 16, 32, 64, 128)},
        {"name": "dropout_rate", "type": "continuous",
         "domain": (0.0, 0.5)},
        {"name": "log_l2", "type": "continuous",
         "domain": (-6, -2)},
        {"name": "batch_size", "type": "discrete",
         "domain": (16, 32, 64, 128)},
    ]

    optimizer = GPyOpt.methods.BayesianOptimization(
        f=objective,
        domain=domain,
        acquisition_type="EI",
        exact_feval=False,
        initial_design_numdata=5,
    )
    optimizer.run_optimization(max_iter=30)

    plot_convergence()
    save_report(optimizer)


if __name__ == "__main__":
    main()
