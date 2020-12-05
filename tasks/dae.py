import boot # noqa: E401
from torch.utils.data import DataLoader
from torch import nn
import torch
from sklearn.metrics import mean_squared_error
import time
import argparse
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import os
import warnings

from utils.misc import (
    time_format,
    Logger,
    EarlyStopping,
    Config as BaseConfig
)
from models.dae import DenoisingAutoencoder, DaeDataset

warnings.filterwarnings('ignore')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')

    if 'get_ipython' in globals():
        return parser.parse_args(args=[])

    return parser.parse_args()


def new_autoencoder(kind, **kwargs):
    if kind == 1:
        return DenoisingAutoencoder(kwargs['n_features'], hidden_size1=2048, hidden_size2=1024)
    if kind == 2:
        return DenoisingAutoencoder(kwargs['n_features'], hidden_size1=2048, hidden_size2=1024,
                                    dropout_ratio0=0.001, dropout_ratio1=0.001, dropout_ratio2=0.001)
    if kind == 3:
        return DenoisingAutoencoder(kwargs['n_features'], hidden_size1=2048, hidden_size2=1024, dropout_ratio2=0.005)


def loop_train(model, criterion, dataloader, optimizer):
    loss = 0.0
    model.train()

    for (x, y) in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        output = model(x)
        loss_ = criterion(output, y)

        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()

        loss += loss_.item() / len(dataloader)

    return loss


def loop_valid(model, criterion, dataloader: DataLoader):
    loss = 0.0

    model.eval()

    preds = []

    with torch.no_grad():
        for (x, y) in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = model(x)
            loss_ = criterion(output, y)
            loss += loss_.item() / len(dataloader)
            preds.append(output)

    return loss, torch.cat(preds, dim=0).cpu().numpy()


def create_pred_feature_df(preds, all_features):
    preds_df = pd.DataFrame(
        np.zeros((preds.shape[0], preds.shape[1] + 1)),
        columns=['sig_id'] + list(range(preds.shape[1]))
    )
    preds_df['sig_id'] = all_features['sig_id']
    preds_df.iloc[:, 1:] = preds

    return preds_df


def run(try_num, config):
    output_dir = f'./dae-out-{try_num}'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    args = get_args()

    train_features = pd.read_csv('../input/lish-moa/train_features.csv')
    test_features = pd.read_csv('../input/lish-moa/test_features.csv')

    if args.debug:
        train_features = train_features.loc[:500]
        config.update(dict(n_epochs=3, n_folds=2))

    all_features = pd.concat([train_features, test_features]).reset_index(drop=True)
    g_features_columns = [col for col in all_features.columns if col.startswith('g-')]
    c_features_columns = [col for col in all_features.columns if col.startswith('c-')]
    feature_columns = g_features_columns + c_features_columns
    n_features = len(feature_columns)

    kfold = MultilabelStratifiedKFold(n_splits=config.n_folds, random_state=42, shuffle=True)
    logger = Logger()

    for fold_index, (train_idx, valid_idx) in enumerate(kfold.split(all_features.values, all_features.values)):
        print('Fold: ', fold_index + 1, flush=True)

        x_train = all_features.loc[train_idx]
        x_valid = all_features.loc[valid_idx]

        model = new_autoencoder(config.model_kind, n_features=n_features).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3, eps=1e-4, verbose=True)
        early_stopping = EarlyStopping(patience=10)
        best_score = np.inf

        for epoch in range(config.n_epochs):
            dataset = DaeDataset(x_train, feature_columns, noise_ratio=config.noise_ratio)
            dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

            train_loss = loop_train(model, criterion, dataloader, optimizer)

            dataset = DaeDataset(x_valid, feature_columns, noise_ratio=config.noise_ratio)
            dataloader = DataLoader(dataset, batch_size=config.valid_batch_size, shuffle=False)
            valid_loss, _ = loop_valid(model, criterion, dataloader)

            scheduler.step(valid_loss)

            logger.update({'fold': fold_index, 'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': valid_loss})
            print(f'epoch {epoch + 1}/{config.n_epochs}  -  train_loss: {train_loss:.5f}  -  ' +
                  f'valid_loss: {valid_loss:.5f}', flush=True)

            if valid_loss < best_score:
                best_score = valid_loss
                torch.save(model.state_dict(), f'./{output_dir}/dae_fold_weight_{fold_index}.pt')

            if early_stopping.should_stop(valid_loss):
                print('Early stopping', flush=True)
                break

    logger.save(f'./{output_dir}/dae_log.csv')
    oof_preds = []

    for fold_index in range(config.n_folds):
        model = new_autoencoder(config.model_kind, n_features=n_features).to(DEVICE)
        model.load_state_dict(torch.load(f'./{output_dir}/dae_fold_weight_{fold_index}.pt'))
        model.eval()

        dataset = DaeDataset(all_features, feature_columns, noise_ratio=config.noise_ratio)
        dataloader = DataLoader(dataset, batch_size=config.valid_batch_size, shuffle=False)

        loss, preds = loop_valid(model, nn.MSELoss(), dataloader)

        logger.update({'fold': fold_index, 'val_loss': loss})
        print('Evaluation   fold: {}  -  valid_loss: {:.5f}'.format(fold_index, loss), flush=True)

        oof_preds.append(preds)

    print('A Whole Evaluation Score: {:.5f}'.format(
        mean_squared_error(all_features.loc[:, feature_columns].values, np.mean(oof_preds, axis=0))
    ), flush=True)

    # for i, preds in enumerate(oof_preds):
    #     create_pred_feature_df(preds, all_features).to_csv(f'./{output_dir}/dae_features_{i}.csv', index=False)
    create_pred_feature_df(
        np.mean(oof_preds, axis=0),
        all_features
    ).to_csv(f'./{output_dir}/dae_features_mean.csv', index=False)


class Config(BaseConfig):
    batch_size = 128
    valid_batch_size = 128 * 4
    learning_rate = 1e-3
    n_epochs = 50
    n_folds = 7
    hidden_size1 = 2048
    hidden_size2 = 1024
    noise_ratio = 0.3
    model_kind = 1


params = [
    dict(learning_rate=1e-4, model_kind=3),
]

for i, p in enumerate(params):
    start_time = time.time()

    config = Config(p)

    print('Start -> Try:', i, flush=True)
    print('config:', config.to_dict(), flush=True)

    run(i, config)

    print('Done -> Try: {}  -  Elapsed: {}'.format(i, time_format(time.time() - start_time)), flush=True)
