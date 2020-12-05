import boot # noqa: E401
import os
import time

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
from utils.misc import (
    set_seed,
    time_format,
    save_pickle,
    load_pickle,
    create_submission,
    Logger,
    Config as BaseConfig
)
from utils.preprocess import (
    normalize,
    add_stats,
    c_squared,
    encode_categorical_features,
    variance_reduction_fit,
    variance_reduction_transform,
    assign_dae_features,
    merge_dae_features
)
from models.loss import get_minority_target_index, mean_log_loss, SmoothBCELoss
from models.models import MoaModel, MoaModelMSDropout, MoaDeeperModel, MoaDeeperModel2
from models.dataset import MoaDataset


warnings.filterwarnings('ignore')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def new_model(model_kind, *args):
    if model_kind == 2:
        return MoaModel(*args, hidden_size1=2048, hidden_size2=2048,
                        output_size=206, dropout_ratio=0.05, f_dropout_ratio=0.05)
    if model_kind == 7:
        return MoaModelMSDropout(*args, hidden_size1=2048, hidden_size2=2048, output_size=206,
                                 dropout_ratio1=0.05, dropout_ratio2=0.05, dropout_ratio3=0.5)
    if model_kind == 10:
        return MoaDeeperModel(*args, hidden_size1=2048, hidden_size2=1024, hidden_size3=512, output_size=206,
                              dropout_ratio1=0.05, dropout_ratio2=0.05, dropout_ratio3=0.05, last_dropout_ratio=0.5)
    if model_kind == 13:
        return MoaDeeperModel2(
            *args,
            hidden_size1=2048,
            hidden_size2=512,
            hidden_size3=512,
            hidden_size4=512,
            output_size=206,
            dropout_ratio1=0.05,
            dropout_ratio2=0.05,
            dropout_ratio3=0.05,
            dropout_ratio4=0.05,
            last_dropout_ratio=0.5)

    raise ValueError()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--only-pred', action='store_true')

    if 'get_ipython' in globals():
        return parser.parse_args(args=[])

    return parser.parse_args()


def loop_train(model, dataloader, optimizer, loss_functions):
    model.train()

    running_loss = 0.0

    train_loss_fun, eval_loss_fun = loss_functions

    for _, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()

        x, y = x.to(DEVICE), y.to(DEVICE)

        preds = model(x)

        metric_loss = eval_loss_fun(preds, y)

        leaning_loss = train_loss_fun(preds, y)
        leaning_loss.backward()
        optimizer.step()

        running_loss += metric_loss.item() / len(dataloader)

    return running_loss


def loop_valid(model, dataloader, eval_loss_fun):
    model.eval()

    running_loss = 0.0

    preds = []

    with torch.no_grad():
        for _, (x, y) in enumerate(dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = eval_loss_fun(logits, y)
            running_loss += loss.item() / len(dataloader)
            preds.append(logits)

    return running_loss, torch.cat(preds, dim=0).cpu().numpy()


def loop_pred(model, dataloader):
    model.eval()
    preds = []

    with torch.no_grad():
        for _, (x, _) in enumerate(dataloader):
            x = x.to(DEVICE)
            logits = model(x)
            preds.append(logits)

    return torch.cat(preds, dim=0).cpu().numpy()


def preprocess(config, model_dir, train_features, train_targets, test_features, dae_features):
    N_ORIGINAL_FEATURES = 872

    g_features_columns = [col for col in train_features.columns if col.startswith('g-')]
    c_features_columns = [col for col in train_features.columns if col.startswith('c-')]

    # Assign DAE features
    if config.dae_strategy == 'replace':
        train_features, test_features = assign_dae_features(
            train_features, test_features, dae_features, N_ORIGINAL_FEATURES)
    else:
        train_features, test_features, _ = merge_dae_features(
            train_features, test_features, dae_features, len(g_features_columns), len(c_features_columns))

    # Drop ctl_vehicle
    train_targets = train_targets.loc[train_features['cp_type'] == 'trt_cp'].reset_index(drop=True)
    train_features = train_features.loc[train_features['cp_type'] == 'trt_cp'].reset_index(drop=True)

    # Categorical encoding
    train_features, test_features, onehot_feature_columns = encode_categorical_features(train_features, test_features)

    # Normalize
    nomalizing_columns = g_features_columns + c_features_columns + onehot_feature_columns
    train_features, test_features = normalize(train_features, test_features, nomalizing_columns,
                                              norm_fun=config.norm_fun, concat_mode=config.norm_concat_mode,
                                              n_quantiles=config.gauss_n_quantiles)

    # Add stats as futures
    feature_groups = [g_features_columns, c_features_columns]
    train_features, test_features, _ = add_stats(train_features, test_features, feature_groups,
                                                 concat_mode=config.stat_concat_mode)

    train_features, test_features, _ = c_squared(train_features, test_features, c_features_columns,
                                                 square_nums=config.square_nums, concat_mode=config.sqrt_concat_mode)

    # Variance encoding
    variance_target_features = list(train_features.iloc[:, 4:].columns)
    pickle_path = f'{model_dir}/variance_encoder.pkl'

    if not os.path.exists(pickle_path):
        vt = variance_reduction_fit(train_features, variance_target_features, config.variance_threshold)
        save_pickle(vt, pickle_path)

    vt = load_pickle(pickle_path)
    train_features = variance_reduction_transform(vt, train_features, variance_target_features)
    test_features = variance_reduction_transform(vt, test_features, variance_target_features)
    print('(variance_reduction) Number of features after applying:', len(train_features.columns), flush=True)

    return train_features, train_targets, test_features


def run(try_num, config):
    logger = Logger()
    args = get_args()

    print('config:', config.to_dict(), flush=True)
    print('args:', args, flush=True)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    model_dir = f'blending-01-nn-{try_num}'

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    train_features = pd.read_csv('../input/lish-moa/train_features.csv')
    train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
    dae_features = pd.read_csv(config.dae_path)
    test_features = pd.read_csv('../input/lish-moa/test_features.csv')

    if args.debug:
        train_features = train_features[:500]
        train_targets = train_targets[:500]
        dae_features = pd.concat([dae_features.iloc[:500], dae_features.iloc[-3982:]]).reset_index(drop=True)

        config.update(dict(
            n_folds=3,
            seeds=[222],
            n_epochs=3,
            batch_size=128,
        ))

    target_columns = [col for col in train_targets.columns if col != 'sig_id']
    n_targets = len(target_columns)

    train_features, train_targets, test_features = preprocess(config, model_dir, train_features,
                                                              train_targets, test_features,
                                                              dae_features)
    features_columns = [col for col in train_features.columns
                        if col not in ['sig_id', 'cp_type', 'cp_time', 'cp_dose',
                                       'cp_type_ctl_vehicle', 'cp_type_trt_cp']]

    metric_loss_function = nn.BCELoss()

    if config.weighted_loss_strategy == 1:
        indices = get_minority_target_index(train_targets, threshold=config.weighted_loss_threshold)
        indices = [int(i not in indices) for i, c in enumerate(target_columns)]
        smooth_loss_function = SmoothBCELoss(smoothing=config.smoothing,
                                             weight=config.weighted_loss_weights,
                                             weight_targets=indices,
                                             n_labels=n_targets)
    else:
        smooth_loss_function = SmoothBCELoss(smoothing=config.smoothing)

    kfold = MultilabelStratifiedKFold(n_splits=config.n_folds, random_state=42, shuffle=True)

    for seed_index, seed in enumerate(config.seeds):
        if args.only_pred:
            print('Skip training', flush=True)
            break

        print(f'Train seed {seed}', flush=True)
        set_seed(seed)

        for fold_index, (train_indices, val_indices) in enumerate(kfold.split(
            train_targets[target_columns].values,
            train_targets[target_columns].values
        )):
            print(f'Train fold {fold_index + 1}', flush=True)

            x_train = train_features.loc[train_indices, features_columns]
            y_train = train_targets.loc[train_indices, target_columns]
            x_val = train_features.loc[val_indices, features_columns]
            y_val = train_targets.loc[val_indices, target_columns]

            model = new_model(config.model_kind, len(features_columns)).to(DEVICE)
            checkpoint_path = f'{model_dir}/repeat-{seed}_Fold-{fold_index + 1}.pt'
            optimizer = optim.Adam(model.parameters(), weight_decay=config.weight_decay, lr=config.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=3, eps=1e-4, verbose=True)

            best_loss = np.inf

            for epoch in range(config.n_epochs):
                dataset = MoaDataset(x_train.values, y_train.values)
                dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

                train_loss = loop_train(model, dataloader, optimizer, loss_functions=(
                    smooth_loss_function,
                    metric_loss_function,
                ))

                dataset = MoaDataset(x_val.values, y_val.values)
                dataloader = DataLoader(dataset, batch_size=config.val_batch_size, shuffle=False)
                valid_loss, _ = loop_valid(model, dataloader, metric_loss_function)

                print(
                    'Epoch {}/{}   -   loss: {:5.5f}   -   val_loss: {:5.5f}'
                    .format(epoch + 1, config.n_epochs, train_loss, valid_loss),
                    flush=True
                )

                logger.update({'epoch': epoch + 1, 'loss': train_loss, 'val_loss': valid_loss})

                scheduler.step(valid_loss)

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    torch.save(model.state_dict(), checkpoint_path)

    oof_preds = np.zeros((len(train_features), len(config.seeds), n_targets))
    test_preds = np.zeros((len(test_features), n_targets))

    for seed_index in range(len(config.seeds)):
        seed = config.seeds[seed_index]

        print(f'Inference for seed {seed}', flush=True)

        _test_preds_in_seed = np.zeros((len(test_features), n_targets))

        for fold_index, (_, valid_indices) in enumerate(kfold.split(
            train_targets[target_columns].values,
            train_targets[target_columns].values
        )):
            x_val = train_features.loc[valid_indices, features_columns]
            y_val = train_targets.loc[valid_indices, target_columns]

            checkpoint_path = f'{model_dir}/repeat-{seed}_Fold-{fold_index + 1}.pt'
            model = new_model(config.model_kind, len(features_columns)).to(DEVICE)
            model.load_state_dict(torch.load(checkpoint_path))

            dataset = MoaDataset(x_val.values, y_val.values)
            dataloader = DataLoader(dataset, batch_size=config.val_batch_size, shuffle=False)
            preds = loop_pred(model, dataloader)

            oof_preds[valid_indices, seed_index, :] = preds

            dataset = MoaDataset(test_features[features_columns].values, None)
            dataloader = DataLoader(dataset, batch_size=config.val_batch_size, shuffle=False)
            preds = loop_pred(model, dataloader)

            _test_preds_in_seed += preds / config.n_folds

        score = mean_log_loss(train_targets.loc[:, target_columns].values,
                              oof_preds[:, seed_index, :], n_targets=n_targets)
        test_preds += _test_preds_in_seed / len(config.seeds)

        print(f'Score for this seed {score:5.5f}', flush=True)
        logger.update({'val_loss': score})

    # Evalucate validation score
    oof_preds = np.mean(oof_preds, axis=1)
    score = mean_log_loss(train_targets.loc[:, target_columns].values, oof_preds, n_targets=n_targets)
    print(f'Overall score is {score:5.5f}', flush=True)

    # Save validation prediction
    oof_pred_df = train_targets.copy()
    oof_pred_df.iloc[:, 1:] = oof_preds
    oof_pred_df.to_csv(f'{model_dir}/oof_pred.csv', index=False)

    # Save log
    logger.update({'val_loss': score})
    logger.save(f'{model_dir}/log.csv')

    # Save Test Prediction
    test_features = pd.read_csv('../input/lish-moa/test_features.csv')
    submission = create_submission(test_features, ['sig_id'] + target_columns)
    submission[target_columns] = test_preds
    submission.loc[test_features['cp_type'] == 'ctl_vehicle', target_columns] = 0
    submission.to_csv(f'{model_dir}/submission.csv', index=False)


class Config(BaseConfig):
    model_kind = 2
    n_folds = 7
    seeds = [222]
    n_epochs = 50
    batch_size = 128
    val_batch_size = 512
    learning_rate = 5e-3
    weight_decay = 1e-5
    n_original_features = 872

    # DAE
    dae_path = 'dae-out-0/dae_features_mean.csv'
    dae_strategy = 'merge'

    # c_squared & Stats
    square_nums = [3]
    stat_concat_mode = False
    sqrt_concat_mode = False

    # Smoothing
    smoothing = 0.001
    smooth_preds = False

    # normalize
    norm_fun = 'rank_gauss',  # ('rank_gauss', 'std', 'minmax')
    norm_concat_mode = False
    gauss_n_quantiles = 100

    # variance
    variance_threshold = 0.8

    # weighted_loss
    weighted_loss_strategy = 1
    weighted_loss_threshold = 30
    weighted_loss_weights = [0.3, 0.2]


params = [
    # basic_nn
    dict(
        model_kind=7,
        learning_rate=5e-3,
        weighted_loss_strategy=None,
        seeds=[1, 10, 13, 16, 19, 20, 333]
    ),
    # deeper_nn
    dict(
        model_kind=10,
        learning_rate=2e-3,
        weighted_loss_strategy=None,
        seeds=[221, 222, 223, 224, 225, 226, 227]
    ),
    # deeper_nn_2
    dict(
        model_kind=13,
        learning_rate=5e-3,
        weighted_loss_strategy=None,
        seeds=[420, 421, 422, 423, 424, 425, 426]
    ),
    # basic_nn_with_weight
    dict(
        model_kind=2,
        learning_rate=5e-3,
        weighted_loss_strategy=1,
        weighted_loss_threshold=40,
        weighted_loss_weights=[4, 3],
        seeds=[170, 171, 172, 173, 174, 175, 176]
    ),
    # deeper_nn_with_weight
    dict(
        model_kind=10,
        learning_rate=5e-3,
        weighted_loss_strategy=1,
        weighted_loss_threshold=40,
        weighted_loss_weights=[4, 3],
        seeds=[120, 121, 122, 123, 124, 125, 126]
    ),
    # deeper_nn_2_with_weight
    dict(
        model_kind=13,
        learning_rate=5e-3,
        weighted_loss_strategy=1,
        weighted_loss_threshold=40,
        weighted_loss_weights=[4, 3],
        seeds=[80, 81, 82, 83, 84, 85, 86]
    ),
]


for (i, p) in enumerate(params):
    print('Start -> Try: {}'.format(i), flush=True)
    start_time = time.time()
    config = Config(p)

    run(i, config)

    print('Done -> Try: {}   Elapsed: {}'.format(i, time_format(time.time() - start_time)), flush=True)
