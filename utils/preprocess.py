import math
import pandas as pd

from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold


def encode_categorical_features(train, test, features=['cp_dose', 'cp_time']):
    adding_columns = []

    for dataset in [train, test]:
        for col in features:
            dummies = pd.get_dummies(dataset[col])

            for val in dummies.columns:
                new_feature = col + '_' + str(val)
                dataset[new_feature] = dummies[val]
                adding_columns.append(new_feature)

    return train, test, adding_columns


def rank_gauss(data_df, feature_columns, n_quantiles=100):
    print('(Rank Gauss) n_quantiles:', n_quantiles, flush=True)

    vec_len = len(data_df)

    for col in (feature_columns):
        raw_vec = data_df[col].values.reshape(vec_len, 1)
        transformer = QuantileTransformer(n_quantiles=n_quantiles, random_state=0, output_distribution="normal")
        data_df[col] = transformer.fit_transform(raw_vec).reshape(1, vec_len)[0]

    return data_df


def standardize(data_df, feature_columns):
    data_df.loc[:, feature_columns] = StandardScaler().fit_transform(data_df.loc[:, feature_columns].values)
    return data_df


def minmax_scaling(data_df, feature_columns):
    data_df.loc[:, feature_columns] = MinMaxScaler().fit_transform(data_df.loc[:, feature_columns].values)
    return data_df


def normalize(train_features, test_features, feature_columns, norm_fun='rank_gauss', concat_mode=False, **kwargs):
    if concat_mode:
        print('(Nomalize) Apply to concatinated dataset', flush=True)
        dataset = [pd.concat([train_features, test_features]).reset_index(drop=True)]
    else:
        print('(Nomalize) Apply to separated (not concatinated) dataset', flush=True)
        dataset = [train_features, test_features]

    for data_df in dataset:
        if norm_fun == 'rank_gauss':
            print('(Nomalize) Apply Rank Gauss', flush=True)
            data_df = rank_gauss(data_df, feature_columns, n_quantiles=kwargs.get('n_quantiles'))
        elif norm_fun == 'std':
            print('(Nomalize) Apply Standardize', flush=True)
            data_df = standardize(data_df, feature_columns)
        else:
            print('(Nomalize) Apply Normalize', flush=True)
            data_df = minmax_scaling(data_df, feature_columns)

    if concat_mode:
        train_features = dataset[0].iloc[:train_features.shape[0]].reset_index(drop=True)
        test_features = dataset[0].iloc[-test_features.shape[0]:].reset_index(drop=True)
    else:
        train_features, test_features = dataset

    return train_features, test_features


def add_stats(train_features, test_features, feature_groups, concat_mode=True):
    if concat_mode:
        print('(add_stats) Apply to concatinated dataset', flush=True)
        dataset = [pd.concat([train_features, test_features]).reset_index(drop=True)]
    else:
        print('(add_stats) Apply to separated (not concatinated) dataset', flush=True)
        dataset = [train_features, test_features]

    before_columns = list(train_features.columns)

    for df in dataset:
        df = pd.concat([train_features, test_features]).reset_index(drop=True)

        for c, features in enumerate(feature_groups):
            df[f'{c}_sum'] = df[features].sum(axis=1)
            df[f'{c}_mean'] = df[features].mean(axis=1)
            df[f'{c}_std'] = df[features].std(axis=1)
            df[f'{c}_kurt'] = df[features].kurtosis(axis=1)
            df[f'{c}_skew'] = df[features].skew(axis=1)
            df[f'{c}_max'] = df[features].max(axis=1)
            df[f'{c}_min'] = df[features].min(axis=1)
            df[f'{c}_25'] = df[features].quantile(0.25, axis=1)
            df[f'{c}_50'] = df[features].quantile(0.50, axis=1)
            df[f'{c}_75'] = df[features].quantile(0.75, axis=1)
            df[f'{c}_var'] = df[features].var(axis=1)

    if concat_mode:
        df = dataset[0]
        train_features = df.iloc[:train_features.shape[0]].reset_index(drop=True)
        test_features = df.iloc[-test_features.shape[0]:].reset_index(drop=True)
    else:
        train_features, test_features = dataset

    added_columns = [c for c in train_features.columns if c not in before_columns]

    return train_features, test_features, added_columns


def c_squared(train_features, test_features, features_c, square_nums=[2], concat_mode=True):
    if concat_mode:
        print('(c_squared) Apply to concatinated dataset', flush=True)
        dataset = [pd.concat([train_features, test_features]).reset_index(drop=True)]
    else:
        print('(c_squared) Apply to separated (not concatinated) dataset', flush=True)
        dataset = [train_features, test_features]

    before_columns = list(train_features.columns)

    for df in dataset:
        for feature in features_c:
            for n in square_nums:
                df[f'{feature}_squared_{n}'] = df[feature] ** n

    if concat_mode:
        df = dataset[0]
        train_features = df.iloc[:train_features.shape[0]].reset_index(drop=True)
        test_features = df.iloc[-test_features.shape[0]:].reset_index(drop=True)
    else:
        train_features, test_features = dataset

    added_columns = [c for c in train_features.columns if c not in before_columns]

    return train_features, test_features, added_columns


def variance_reduction_fit(train_features, features, threshold=0.8):
    no_targets = [c for c in train_features.columns if c not in features]

    print('(variance_reduction) threshold:', threshold)
    print('(variance_reduction) Number of targets:', len(features), flush=True)
    print('(variance_reduction) Number of non-targets:', len(no_targets), flush=True)

    vt = VarianceThreshold(threshold).fit(train_features.loc[:, features])

    return vt


def variance_reduction_transform(vt, data, features):
    no_targets = [c for c in data.columns if c not in features]

    print('(variance_reduction) Number of targets:', len(features), flush=True)
    print('(variance_reduction) Number of non-targets:', len(no_targets), flush=True)

    data_transformed = vt.transform(data.loc[:, features])
    data = pd.concat([data.loc[:, no_targets], pd.DataFrame(data_transformed)], axis=1)

    return data


def assign_dae_features(train_features, test_features, dae_features, n_features):
    train_features.iloc[:, -n_features:] = dae_features.iloc[:len(train_features), -n_features:].values
    test_features.iloc[:, -n_features:] = dae_features.iloc[len(train_features):, -n_features:].values
    return train_features, test_features


def merge_dae_features(train_features, test_features, dae_features, n_features_g, n_features_c):
    column_map = {str(i): f'dae-{i}-g' for i in range(n_features_g)}
    column_map.update({str(i + n_features_g): f'dae-{i}-c' for i in range(n_features_c)})
    adding_columns = list(column_map.values())

    dae_features = dae_features.rename(columns=column_map)
    all_features = pd.concat([train_features, test_features]).reset_index(drop=True)
    joined_df = all_features.set_index('sig_id').join(dae_features.set_index('sig_id')).reset_index()

    train_features = joined_df.iloc[:len(train_features)].reset_index(drop=True)
    test_features = joined_df.iloc[len(train_features):].reset_index(drop=True)

    return train_features, test_features, adding_columns


def apply_pca(train_features, test_features, feature_groups, n_comp_ratio=0.3, concat_mode=True):
    if concat_mode:
        print('(PCA) Apply to concatinated dataset', flush=True)
        dataset = [pd.concat([train_features, test_features]).reset_index(drop=True)]
    else:
        print('(PCA) Apply to separated (not concatinated) dataset', flush=True)
        dataset = [train_features, test_features]

    for add_df in dataset:
        add_df = pd.concat([train_features, test_features]).reset_index(drop=True)
        adding_columns = []

        for c, features in enumerate(feature_groups):
            n_comp = math.ceil(len(features) * n_comp_ratio)
            pca_data = PCA(n_components=n_comp, random_state=42).fit_transform(add_df[features].values)

            for i in range(n_comp):
                new_column = f'pca-{c}-{i}'
                add_df[new_column] = pca_data[:, i]
                adding_columns.append(new_column)

    if concat_mode:
        train_features = dataset[0][:train_features.shape[0]].reset_index(drop=True)
        test_features = dataset[0][-test_features.shape[0]:].reset_index(drop=True)
    else:
        train_features, test_features = dataset

    return train_features, test_features, adding_columns
