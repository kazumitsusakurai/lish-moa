import boot  # noqa: E401
import warnings
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor
import torch.optim as optim
import argparse
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import os
import time

from utils.misc import (
    set_seed,
    time_format,
    save_pickle,
    load_pickle,
    create_submission,
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
    merge_dae_features,
    apply_pca
)
from models.loss import mean_log_loss, sigmoid, SmoothBCEwLogits

warnings.filterwarnings('ignore')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--only-pred', action='store_true')

    if 'get_ipython' in globals():
        return parser.parse_args(args=[])

    return parser.parse_args()


class LogitsLogLoss(Metric):
    """
    LogLoss with sigmoid applied
    """

    def __init__(self):
        self._name = "logits_ll"
        self._maximize = False

    def __call__(self, y_true, y_pred):
        """
        Compute LogLoss of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_score: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            LogLoss of predictions vs targets.
        """
        logits = 1 / (1 + np.exp(-y_pred))
        aux = (1 - y_true) * np.log(1 - logits + 1e-15) + y_true * np.log(logits + 1e-15)
        return np.mean(-aux)


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

    # Grouping features
    feature_groups = [g_features_columns, c_features_columns]

    # Add stats as futures
    train_features, test_features, _ = add_stats(train_features, test_features, feature_groups,
                                                 concat_mode=config.stat_concat_mode)

    train_features, test_features, _ = c_squared(train_features, test_features, c_features_columns,
                                                 square_nums=config.square_nums, concat_mode=config.sqrt_concat_mode)

    # PCA
    feature_names_pca = []
    if config.skip_pca is False:
        train_features, test_features, feature_names_pca = apply_pca(train_features, test_features,
                                                                     feature_groups=feature_groups,
                                                                     n_comp_ratio=config.pca_n_comp_ratio,
                                                                     concat_mode=config.pca_concat_mode)
        print(
            f'(PCA) Adding {len(feature_names_pca)} features ' +
            f'and having a total of {len(train_features.columns)} features.',
            flush=True
        )
        print('(PCA) train:', train_features.shape, flush=True)
        print('(PCA) test:', test_features.shape, flush=True)

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
    args = get_args()

    print('config:', config.to_dict(), flush=True)
    print('args:', args, flush=True)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    model_dir = f'blending-02-tabnet-{try_num}'

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

    train_features = train_features[features_columns]
    test_features = test_features[features_columns]

    smooth_loss_function = SmoothBCEwLogits(smoothing=config.smoothing)
    kfold = MultilabelStratifiedKFold(n_splits=config.n_folds, random_state=42, shuffle=True)

    oof_preds = np.zeros((len(train_features), len(config.seeds), n_targets))
    test_preds = []

    for seed_index, seed in enumerate(config.seeds):
        print(f'Train seed {seed}', flush=True)
        set_seed(seed)

        for fold_index, (train_indices, val_indices) in enumerate(kfold.split(
            train_targets[target_columns].values,
            train_targets[target_columns].values
        )):
            print(f'Train fold {fold_index + 1}', flush=True)
            x_train = train_features.loc[train_indices, features_columns].values
            y_train = train_targets.loc[train_indices, target_columns].values
            x_val = train_features.loc[val_indices, features_columns].values
            y_val = train_targets.loc[val_indices, target_columns].values

            weights_path = f'{model_dir}/weights-{seed}-{fold_index}.pt'

            tabnet_conf = dict(
                seed=seed,
                optimizer_fn=optim.Adam,
                scheduler_fn=optim.lr_scheduler.ReduceLROnPlateau,
                n_d=32,
                n_a=32,
                n_steps=1,
                gamma=1.3,
                lambda_sparse=0,
                momentum=0.02,
                optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                scheduler_params=dict(mode="min", patience=5, min_lr=1e-5, factor=0.9),
                mask_type="entmax",
                verbose=10,
                n_independent=1,
                n_shared=1,
            )

            if args.only_pred:
                print('Skip training', flush=True)
            else:
                model = TabNetRegressor(**tabnet_conf)

                model.fit(
                    X_train=x_train,
                    y_train=y_train,
                    eval_set=[(x_val, y_val)],
                    eval_name=['val'],
                    eval_metric=['logits_ll'],
                    max_epochs=config.n_epochs,
                    patience=20,
                    batch_size=1024,
                    virtual_batch_size=32,
                    num_workers=1,
                    drop_last=True,
                    loss_fn=smooth_loss_function
                )

                model.save_model(weights_path)
                print('Save weights to: ', weights_path, flush=True)

            model = TabNetRegressor(**tabnet_conf)
            model.load_model(f'{weights_path}.zip')

            val_preds = sigmoid(model.predict(x_val))
            score = mean_log_loss(y_val, val_preds, n_targets)
            print(f'fold_index {fold_index}   -   val_loss: {score:5.5f}', flush=True)

            oof_preds[val_indices, seed_index, :] = val_preds

            preds = sigmoid(model.predict(test_features.values))
            test_preds.append(preds)

        score = mean_log_loss(train_targets[target_columns].values, oof_preds[:, seed_index, :], n_targets)
        print(f'Seed {seed}   -   val_loss: {score:5.5f}', flush=True)

    oof_preds = np.mean(oof_preds, axis=1)
    score = mean_log_loss(train_targets[target_columns].values, oof_preds, n_targets)
    print(f'Overall score is {score:5.5f}', flush=True)

    oof_pred_df = train_targets.copy()
    oof_pred_df.loc[:, target_columns] = oof_preds
    oof_pred_df.to_csv(f'{model_dir}/oof_pred.csv', index=False)

    test_features = pd.read_csv('../input/lish-moa/test_features.csv')
    submission = create_submission(test_features, ['sig_id'] + target_columns)
    submission[target_columns] = np.mean(test_preds, axis=0)
    submission.loc[test_features['cp_type'] == 'ctl_vehicle', target_columns] = 0
    submission.to_csv(f'{model_dir}/submission.csv', index=False)


class Config(BaseConfig):
    n_epochs = 200
    seeds = [3, 11, 23]

    # DAE
    dae_path = 'dae-out-0/dae_features_mean.csv'
    dae_strategy = 'merge'

    # c_squared & Stats
    square_nums = [3]
    stat_concat_mode = False
    sqrt_concat_mode = False

    # Smoothing
    smoothing = 0.001

    # PCA
    skip_pca = False
    pca_n_comp_ratio = 0.12970
    pca_concat_mode = False

    # normalize
    norm_fun = 'rank_gauss',  # ('rank_gauss', 'std', 'minmax')
    norm_concat_mode = False
    gauss_n_quantiles = 100

    # variance
    variance_threshold = 0.8


params = [
    dict(seeds=[3, 11, 23, 31, 32, 33]),
]

for (i, p) in enumerate(params):
    print('Start -> Try: {}'.format(i), flush=True)
    start_time = time.time()

    config = Config(p)

    run(i, config)

    print('Done -> Try: {}   Elapsed: {}'.format(i, time_format(time.time() - start_time)), flush=True)
