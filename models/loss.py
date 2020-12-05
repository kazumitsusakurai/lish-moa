
import pandas as pd
import numpy as np

from sklearn.metrics import log_loss

import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mean_log_loss(y_true, y_pred, n_targets=206):
    metrics = []

    for i in range(n_targets):
        metrics.append(log_loss(y_true[:, i], y_pred[:, i].astype(float), labels=[0, 1]))

    return np.mean(metrics)


def get_minority_target_index(train_targets, threshold=30):
    target_sum = train_targets.iloc[:, 1:].sum()

    df = pd.DataFrame({
        'name': target_sum.index.values,
        'amount': target_sum.values,
        'rate': target_sum.values / target_sum.sum(),
    })

    minorities = list(df[df.amount <= threshold].name.values)

    return [i for i, c in enumerate(train_targets.columns) if c in minorities]


def apply_smoothing(value, smoothing):
    return value * (1.0 - smoothing) + (smoothing * 0.5)


def bce_loss(logits, y):
    return F.binary_cross_entropy_with_logits(logits, y, reduction="mean")


class SmoothBCELoss(nn.BCELoss):
    def __init__(self, smoothing=0.001, weight=None, weight_targets=None, n_labels=None):
        if weight is not None:
            weight = torch.Tensor([weight[weight_targets[i]] for i in range(n_labels)]).to(DEVICE)

        super().__init__(weight=weight)
        self.smoothing = smoothing

    @staticmethod
    def _smooth(labels, smoothing=0.0):
        assert 0 <= smoothing < 1

        with torch.no_grad():
            labels = apply_smoothing(labels, smoothing)

        return labels

    def forward(self, preds, labels):
        labels = SmoothBCELoss._smooth(labels, self.smoothing)

        loss = super().forward(preds, labels)

        return loss


class SmoothBCEwLogits(nn.BCEWithLogitsLoss):
    def __init__(self, smoothing=0.001, weight=None, weight_targets=None, n_labels=None):
        if weight is not None:
            weight = torch.Tensor([weight[weight_targets[i]] for i in range(n_labels)]).to(DEVICE)

        super().__init__(weight=weight)
        self.smoothing = smoothing

    @staticmethod
    def _smooth(labels, smoothing=0.0):
        assert 0 <= smoothing < 1

        with torch.no_grad():
            labels = apply_smoothing(labels, smoothing)

        return labels

    def forward(self, preds, labels):
        labels = SmoothBCEwLogits._smooth(labels, self.smoothing)

        loss = super().forward(preds, labels)

        return loss
