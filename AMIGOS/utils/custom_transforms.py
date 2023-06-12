import numpy as np
import itertools
import torch
from torch import nn


class AnnotatorsAverage(object):
    def __call__(self, x: torch.Tensor):
        return torch.mean(x, dim=1)


class AnnotatorsAverageClean(object):
    def __call__(self, x: torch.Tensor):
        avg = torch.mean(x, dim=1).unsqueeze(1)
        dist = torch.sub(x, avg)
        idx = torch.argmax(dist, dim=1)

        y = [torch.cat((x[i][:idx[i]], x[i][idx[i]+1:])) for i in range(x.size(0))]
        y = torch.vstack(y)
        y = torch.mean(y, dim=1)
        return y


class AnnotatorsDistance(object):
    def __call__(self, x: torch.Tensor):
        annot = x.size(1)
        perm = itertools.combinations(range(annot), 2)
        darray = [torch.abs(torch.sub(x[:, p[0]], x[:, p[1]])) for p in perm]
        darray = torch.vstack(darray)
        darray = torch.mean(darray, dim=0)
        return darray


class AnnotatorsVar(object):
    def __call__(self, x: torch.Tensor):
        return torch.var(x, dim=1)


class AnnotatorsVarMean(object):
    def __call__(self, x: torch.Tensor):
        var, mean = torch.var_mean(x, dim=1)
        lab = torch.stack([var, mean])
        return lab


class ColumnSelect(object):
    def __init__(self, keys: list):
        self.keys = keys

    def __call__(self, feats: dict):
        feats = np.array([feats.get(key) for key in self.keys])
        return feats


class FlattenLabels(object):
    def __call__(self, x: torch.Tensor):
        return x.reshape(-1)


def images_collate(batch):
    x_data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.stack(target, 0)
    x_data = nn.utils.rnn.pad_sequence(x_data, batch_first=True)
    x_data = x_data.permute(0, 2, 1, 3, 4)
    return x_data, target


def series_collate(batch):
    x_data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.stack(target, 0)
    x_data = nn.utils.rnn.pad_sequence(x_data, batch_first=True)
    x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], x_data.shape[2])
    return x_data, target
