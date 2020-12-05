# Reference: https://www.kaggle.com/markpeng/deepinsight-efficientnet-b3-noisystudent

import boot # noqa: E401
import os
import gc
import warnings
import argparse
import time
import pandas as pd
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils.misc import (
    set_seed,
    time_format,
    save_pickle,
    load_pickle,
    Logger,
    EarlyStopping,
    Config as BaseConfig
)
from utils.preprocess import (
    normalize,
    variance_reduction_fit,
    variance_reduction_transform,
    assign_dae_features,
    merge_dae_features
)
from models.loss import get_minority_target_index, mean_log_loss, bce_loss, SmoothBCEwLogits
from models.deepingsight import (
    DeepInsightTransformer,
    LogScaler,
    MoAImageSwapDataset,
    MoAImageDataset,
    TestDataset,
    MoAEfficientNet,
)


warnings.filterwarnings('ignore')
gc.enable()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--only-pred', action='store_true')
    parser.add_argument('--return-first-fold', action='store_true')

    if 'get_ipython' in globals():
        return parser.parse_args(args=[])

    return parser.parse_args()


def loop_train(model, criterion, dataloader, optimizer):
    loss = 0.0
    model.train()
    total_iter = len(dataloader)

    for i, (x, y) in enumerate(dataloader):
        if i > 0 and i % 15 == 0:
            print(f'iter: {i} / {total_iter}  -  loss: {loss:.6f}', flush=True)

        x, y = x.to(DEVICE).float(), y.to(DEVICE).float()

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
            x, y = x.to(DEVICE).float(), y.to(DEVICE).float()
            output = model(x)
            loss_ = criterion(output, y)
            loss += loss_.item() / len(dataloader)
            preds.append(F.sigmoid(output))

    return loss, torch.cat(preds, dim=0).cpu().numpy()


def loop_preds(model, dataloader: DataLoader):
    model.eval()
    preds = []

    with torch.no_grad():
        for (x, _) in dataloader:
            x = x.to(DEVICE).float()
            output = model(x)
            output = F.sigmoid(output)
            preds.append(output)

    return torch.cat(preds, dim=0).cpu().numpy()


def run(try_num, config):
    args = get_args()

    print('args', args, flush=True)
    print('config:', config.to_dict(), flush=True)

    set_seed(config.rand_seed)

    pretrained_model = f"tf_efficientnet_b3_ns"
    model_dir = f'deepinsight-{try_num}'

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    train_features = pd.read_csv(f"../input/lish-moa/train_features.csv")
    train_targets = pd.read_csv(f"../input/lish-moa/train_targets_scored.csv")
    test_features = pd.read_csv(f"../input/lish-moa/test_features.csv")

    if config.dae_path:
        dae_features = pd.read_csv(config.dae_path)

    if args.debug:
        train_features = train_features.iloc[:500]
        train_targets = train_targets.iloc[:500]
        if config.dae_path:
            dae_features = pd.concat([dae_features.iloc[:500], dae_features.iloc[-3982:]]).reset_index(drop=True)

        config.update(dict(
            kfolds=3,
            n_epoch=3
        ))

    train_features = train_features.sort_values(by=["sig_id"], axis=0, inplace=False).reset_index(drop=True)
    train_targets = train_targets.sort_values(by=["sig_id"], axis=0, inplace=False).reset_index(drop=True)

    cat_features_columns = ["cp_dose", 'cp_time']
    num_feature_columns = [c for c in train_features.columns
                           if c != "sig_id" and c not in cat_features_columns + ['cp_type']]
    all_features_columns = cat_features_columns + num_feature_columns
    target_columns = [c for c in train_targets.columns if c != "sig_id"]
    g_feature_columns = [c for c in num_feature_columns if c.startswith("g-")]
    c_feature_columns = [c for c in num_feature_columns if c.startswith("c-")]

    if config.dae_path:
        if config.dae_strategy == 'replace':
            train_features, test_features = assign_dae_features(
                train_features, test_features, dae_features, len(num_feature_columns))
        else:
            train_features, test_features, dae_feature_columns = merge_dae_features(
                train_features, test_features, dae_features, len(g_feature_columns), len(c_feature_columns))
            all_features_columns += dae_feature_columns

    train_targets = train_targets.loc[train_features['cp_type'] == 'trt_cp'].reset_index(drop=True)
    train_features = train_features.loc[train_features['cp_type'] == 'trt_cp'].reset_index(drop=True)

    if config.normalizer == 'rank':
        train_features, test_features = normalize(train_features, test_features, num_feature_columns)

    for df in [train_features, test_features]:
        df['cp_type'] = df['cp_type'].map({'ctl_vehicle': 0, 'trt_cp': 1})
        df['cp_dose'] = df['cp_dose'].map({'D1': 0, 'D2': 1})
        df['cp_time'] = df['cp_time'].map({24: 0, 48: 0.5, 72: 1})

    if config.variance_target_type == 1:
        pickle_path = f'{model_dir}/variance_reduction.pkl'

        variance_target_features = num_feature_columns
        if config.dae_path and config.dae_strategy != 'replace':
            variance_target_features += dae_feature_columns

        if not os.path.exists(pickle_path):
            vt = variance_reduction_fit(train_features, variance_target_features, config.variance_threshold)
            save_pickle(vt, pickle_path)

        vt = load_pickle(pickle_path)
        train_features = variance_reduction_transform(vt, train_features, variance_target_features)
        test_features = variance_reduction_transform(vt, test_features, variance_target_features)
        print('(variance_reduction) Number of features after applying:', len(train_features.columns), flush=True)
        all_features_columns = list(train_features.columns[1:])

    skf = MultilabelStratifiedKFold(n_splits=config.kfolds, shuffle=True, random_state=config.rand_seed)
    y_labels = np.sum(train_targets.drop("sig_id", axis=1), axis=0).index.tolist()
    logger = Logger()

    for fold_index, (train_index, val_index) in enumerate(skf.split(train_features, train_targets[y_labels])):
        if args.only_pred:
            print('Skip training', flush=True)
            break

        print(f'Fold: {fold_index}', train_index.shape, val_index.shape, flush=True)

        X_train = train_features.loc[train_index, all_features_columns].copy().values
        y_train = train_targets.iloc[train_index, 1:].copy().values
        X_valid = train_features.loc[val_index, all_features_columns].copy().values
        y_valid = train_targets.iloc[val_index, 1:].copy().values

        if config.normalizer == 'log':
            scaler = LogScaler()
            if config.norm_apply_all:
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_valid = scaler.transform(X_valid)
            else:
                target_features = [i for i, c in enumerate(all_features_columns) if c in num_feature_columns]
                non_target_features = [i for i, c in enumerate(all_features_columns) if c not in num_feature_columns]

                scaler.fit(X_train[:, target_features])
                X_train_tr = scaler.transform(X_train[:, target_features])
                X_valid_tr = scaler.transform(X_valid[:, target_features])
                X_train = np.concatenate([X_train[:, non_target_features], X_train_tr], axis=1)
                X_valid = np.concatenate([X_valid[:, non_target_features], X_valid_tr], axis=1)
            save_pickle(scaler, f'{model_dir}/scaler-{fold_index}.pkl')

        transformer = DeepInsightTransformer(
            feature_extractor=config.extractor,
            pixels=config.resolution,
            perplexity=config.perplexity,
            random_state=config.rand_seed,
            n_jobs=-1
        ).fit(X_train)

        save_pickle(transformer, f'{model_dir}/transformer-{fold_index}.pkl')

        model = MoAEfficientNet(
            pretrained_model_name=pretrained_model,
            fc_size=config.fc_size,
            drop_rate=config.drop_rate,
            drop_connect_rate=config.drop_connect_rate,
            weight_init='goog',
        ).to(DEVICE)

        if config.smoothing is not None:
            if config.weighted_loss_weights is not None:
                indices = get_minority_target_index(train_targets, threshold=config.weighted_loss_threshold)
                indices = [int(i not in indices) for i, c in enumerate(target_columns)]
                train_loss_function = SmoothBCEwLogits(
                    smoothing=config.smoothing,
                    weight=config.weighted_loss_weights,
                    weight_targets=indices,
                    n_labels=len(target_columns))
            else:
                train_loss_function = SmoothBCEwLogits(smoothing=config.smoothing)
        else:
            train_loss_function = bce_loss

        eval_loss_function = bce_loss

        optimizer = optim.Adam(model.parameters(), weight_decay=config.weight_decay, lr=config.learning_rate)

        if config.scheduler_type == 'ca':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.t_max, eta_min=0, last_epoch=-1)
        elif config.scheduler_type == 'ms':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.ms_scheduler_milestones, gamma=0.1)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=config.rp_patience, eps=1e-4, verbose=True)

        early_stopping = EarlyStopping(patience=7)
        best_score = np.inf
        start_time = time.time()

        for epoch in range(config.n_epoch):

            if config.swap_enable:
                dataset = MoAImageSwapDataset(
                    X_train,
                    y_train,
                    transformer,
                    image_size=config.image_size,
                    swap_prob=config.swap_prob,
                    swap_portion=config.swap_portion)
            else:
                dataset = MoAImageDataset(X_train, y_train, transformer, image_size=config.image_size)

            dataloader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
                drop_last=False)
            loss = loop_train(model, train_loss_function, dataloader, optimizer)

            if config.scheduler_type == 'rp':
                scheduler.step(loss)
            else:
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print('current learning rate:', param_group['lr'])

            del dataset, dataloader

            dataset = MoAImageDataset(X_valid, y_valid, transformer, image_size=config.image_size)
            dataloader = DataLoader(
                dataset,
                batch_size=config.infer_batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                drop_last=False)
            valid_loss, valid_preds = loop_valid(model, eval_loss_function, dataloader)

            del dataset, dataloader

            logger.update({'fold': fold_index, 'epoch': epoch + 1, 'train_loss': loss, 'val_loss': valid_loss})
            print(f'epoch {epoch + 1}/{config.n_epoch}  -  train_loss: {loss:.5f}  -  ' +
                  f'valid_loss: {valid_loss:.5f}  -  elapsed: {time_format(time.time() - start_time)}', flush=True)

            if valid_loss < best_score:
                best_score = valid_loss
                torch.save(model.state_dict(), f'./{model_dir}/deepinsight-{fold_index}.pt')

            if early_stopping.should_stop(valid_loss):
                print('Early stopping', flush=True)
                break

        print(f'Done -> Fold {fold_index}/{config.kfolds}  -  best_valid_loss: {best_score:.5f}  -  ' +
              f'elapsed: {time_format(time.time() - start_time)}', flush=True)

        torch.cuda.empty_cache()
        gc.collect()

        if args.return_first_fold:
            logger.save(f'{model_dir}/log.csv')
            return

    test_preds = np.zeros((test_features.shape[0], len(target_columns)))
    start_time = time.time()
    print('Start infarence', flush=True)

    oof_preds = np.zeros((len(train_features), len(target_columns)))
    eval_loss_function = bce_loss

    for fold_index, (train_index, val_index) in enumerate(skf.split(train_features, train_targets[y_labels])):
        print(f'Infarence Fold: {fold_index}', train_index.shape, val_index.shape, flush=True)
        X_valid = train_features.loc[val_index, all_features_columns].copy().values
        y_valid = train_targets.iloc[val_index, 1:].copy().values
        X_test = test_features[all_features_columns].values

        if config.normalizer == 'log':
            scaler = load_pickle(f'{model_dir}/scaler-{fold_index}.pkl')
            X_valid = scaler.transform(X_valid)
            X_test = scaler.transform(X_test)

        transformer = load_pickle(f'{model_dir}/transformer-{fold_index}.pkl')
        model = MoAEfficientNet(
            pretrained_model_name=pretrained_model,
            fc_size=config.fc_size,
            drop_rate=config.drop_rate,
            drop_connect_rate=config.drop_connect_rate,
            weight_init='goog',
        ).to(DEVICE)
        model.load_state_dict(torch.load(f'./{model_dir}/deepinsight-{fold_index}.pt'))

        dataset = MoAImageDataset(X_valid, y_valid, transformer, image_size=config.image_size)
        dataloader = DataLoader(
            dataset,
            batch_size=config.infer_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False)
        valid_loss, valid_preds = loop_valid(model, eval_loss_function, dataloader)
        print(f'Fold {fold_index}/{config.kfolds}  -  fold_valid_loss: {valid_loss:.5f}', flush=True)
        logger.update({'fold': fold_index, 'val_loss': valid_loss})

        oof_preds[val_index, :] = valid_preds

        dataset = TestDataset(X_test, None, transformer, image_size=config.image_size)
        dataloader = DataLoader(
            dataset,
            batch_size=config.infer_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False)

        preds = loop_preds(model, dataloader)
        test_preds += preds / config.kfolds

    oof_preds_df = train_targets.copy()
    oof_preds_df.loc[:, target_columns] = oof_preds.clip(0, 1)
    oof_preds_df.to_csv(f'{model_dir}/oof_preds.csv', index=False)
    oof_loss = mean_log_loss(train_targets.loc[:, target_columns].values, oof_preds)

    print(f'OOF Validation Loss: {oof_loss:.6f}', flush=True)
    print(f'Done infarence  Elapsed {time_format(time.time() - start_time)}', flush=True)
    logger.update({'fold': 'oof', 'val_loss': oof_loss})
    logger.save(f'{model_dir}/log.csv')

    submission = pd.DataFrame(data=test_features['sig_id'].values, columns=['sig_id'])
    submission = submission.reindex(columns=['sig_id'] + target_columns)
    submission.loc[:, target_columns] = test_preds.clip(0, 1)
    submission.loc[test_features['cp_type'] == 0, submission.columns[1:]] = 0
    submission.to_csv(f'{model_dir}/submission.csv', index=False)


class Config(BaseConfig):
    rand_seed = 222
    learning_rate = 5e-4
    weight_decay = 1e-5

    drop_connect_rate = 0.2
    fc_size = 512
    smoothing = 0.001
    normalizer = 'log'  # log, rank
    norm_apply_all = True

    # scheduler
    t_max = 5
    scheduler_type = 'rp'
    rp_patience = 3
    ms_scheduler_milestones = [10, 25]

    # DAE
    dae_path = 'dae-out-0/dae_features_mean.csv'
    dae_strategy = 'merge'

    # Swap Noise
    swap_prob = 0.15
    swap_portion = 0.1
    swap_enable = True

    # Transformer
    extractor = 'tsne'
    perplexity = 20

    batch_size = 42
    infer_batch_size = 128
    image_size = 300  # B3
    drop_rate = 0.3  # B3
    resolution = 300

    # variance reduction
    variance_threshold = 0.8
    # 1: original numerical features, 2: original and pca, etc. 3: all
    variance_target_type = 1

    kfolds = 10
    n_epoch = 50

    weighted_loss_weights = None
    weighted_loss_threshold = None


params = [
    # dict(
    #     rp_patience=2,
    #     scheduler_type='rp',
    #     rand_seed=1120,
    # ),
    # dict(
    #     scheduler_type='ms',
    #     ms_scheduler_milestones=[10, 25],
    #     rand_seed=1121
    # ),
    # dict(
    #     scheduler_type='ca',
    #     t_max=15,
    #     rand_seed=1122
    # ),
    # dict(
    #     scheduler_type='ca',
    #     t_max=15,
    #     rand_seed=1123
    # ),
    dict(
        scheduler_type='ca',
        t_max=15,
        weighted_loss_threshold=40,
        weighted_loss_weights=[4, 3],
        rand_seed=1124
    ),
    dict(
        scheduler_type='ms',
        ms_scheduler_milestones=[15, 30],
        weighted_loss_threshold=40,
        weighted_loss_weights=[4, 3],
        rand_seed=1125
    ),
]

for i, p in enumerate(params):
    start_time = time.time()
    print('Start -> Try: {}'.format(i), flush=True)
    config = Config(p)
    run(i, config)
    print('Done -> Try: {}   Elapsed: {}'.format(i, time_format(time.time() - start_time)), flush=True)
