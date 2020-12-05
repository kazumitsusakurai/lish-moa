import random
import os
import numpy as np
import pandas as pd
import torch
import pickle
from pickle import dump, load


def time_format(time_sec, n_round=2):
    hours = int(time_sec // (60 * 60))

    if hours != 0:
        minutes = int((time_sec % (hours * 60 * 60)) // 60)
    else:
        minutes = int(time_sec // 60)
    if hours != 0 and minutes != 0:
        seconds = round((time_sec % (hours * 60 * 60)) % (minutes * 60), n_round)
    elif hours != 0:
        seconds = round(time_sec % (hours * 60 * 60), n_round)
    elif minutes != 0:
        seconds = round(time_sec % (minutes * 60), n_round)
    else:
        seconds = round(time_sec, n_round)

    hours = str(hours)

    if minutes < 10:
        minutes = '0%s' % str(minutes)
    else:
        minutes = str(minutes)
    if seconds < 10:
        seconds = '0%s' % str(seconds)
    else:
        seconds = str(seconds)
    return f'{hours}:{minutes}:{seconds}'


def save_pickle(obj, path):
    dump(obj, open(path, 'wb'),
         pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    return load(open(path, 'rb'))


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Logger:
    def __init__(self):
        self.log = []

    def update(self, row):
        self.log.append(row)

    def save(self, path):
        pd.DataFrame(self.log).to_csv(path, index=False)


class EarlyStopping:
    def __init__(self, patience, mode='min'):
        if mode == 'min':
            self.last_score = np.inf
        else:
            self.last_score = -np.inf

        self.patience = patience
        self.counter = 0
        self.mode = mode

    def should_stop(self, score):
        if self.mode == 'max':
            evaluation = self.last_score < score
        else:
            evaluation = self.last_score > score

        if evaluation is not True:
            self.counter += 1
        else:
            self.counter = 0

        self.last_score = score

        return self.counter > self.patience


class Config:
    def __init__(self, params={}):
        self.update(params)

    def update(self, params={}):
        for key, value in params.items():
            setattr(self, key, value)

    def to_dict(self):
        return {i: getattr(self, i) for i in dir(self)
                if i.find('__') != 0 and not callable(getattr(self, i))}


def create_submission(test, train_target_columns):
    submission = pd.DataFrame(columns=train_target_columns)
    submission.sig_id = test.sig_id
    submission.iloc[:, 1:] = 0.0

    return submission
