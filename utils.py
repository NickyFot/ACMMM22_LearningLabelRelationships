import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt


class IgnoreFiles(object):
    def __init__(self, pattern):
        self.pattern = pattern

    def __call__(self, clip):
        if not isinstance(clip, np.ndarray):
            clip = np.array(clip, dtype=object)
        idx = [self.pattern not in frame for frame in clip]
        return clip[idx]


class TemporalDownSample(object):
    def __init__(self, factor: int):
        self.factor = factor

    def __call__(self, clip: torch.tensor):
        if not isinstance(clip, np.ndarray):
            clip = np.array(clip, dtype=object)
        idx = range(len(clip))
        idx = [(idi % self.factor) == 0 for idi in idx]
        return clip[idx]


class RandomSequence(object):
    def __init__(self, seq_size):
        self.seq_size = seq_size

    def __call__(self, clip: torch.tensor):
        rnd_start = torch.randint(len(clip), (1,))
        end_idx = rnd_start+self.seq_size
        if end_idx < len(clip):
            new_clip = clip[rnd_start:end_idx]
        else:
            end_idx -= len(clip)
            new_clip = np.concatenate((clip[rnd_start:], clip[:end_idx]))
        if len(new_clip) < self.seq_size:
            pad = self.seq_size - len(new_clip)
            new_clip = np.pad(new_clip, (0, pad), 'reflect')
        return new_clip


class FrameSequence(object):
    def __init__(self, seq_size):
        self.seq_size = seq_size

    def __call__(self, clip: torch.tensor, start_idx: int):
        end_idx = start_idx+self.seq_size
        if end_idx < len(clip):
            return clip[start_idx:end_idx]
        else:
            end_idx -= len(clip)
            new_clip = torch.vstack((clip[start_idx:], clip[:end_idx]))
            return new_clip


class NormArousal(object): # OMG dataset only
    def __init__(self, arr_idx):
        self.idx = arr_idx

    def __call__(self, labels):
        labels[self.idx] = 2 * labels[self.idx] - 1
        return labels


def PCC(a: torch.tensor, b: torch.tensor):
    am = torch.mean(a, dim=0)
    bm = torch.mean(b, dim=0)
    num = torch.sum((a - am) * (b - bm), dim=0)
    den = torch.sqrt(sum((a - am) ** 2) * sum((b - bm) ** 2)) + 1e-5
    return num/den


def CCC(a: torch.tensor, b: torch.tensor):
    rho = 2 * PCC(a,b) * a.std(dim=0, unbiased=False) * b.std(dim=0, unbiased=False)
    rho /= (a.var(dim=0, unbiased=False) + b.var(dim=0, unbiased=False) + torch.pow(a.mean(dim=0) - b.mean(dim=0), 2) + 1e-5)
    return rho


def info_recon_loss(features, labels):
    labels = F.normalize(labels)
    labels = torch.matmul(labels, labels.T)

    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    return similarity_matrix, labels

class NormLabels(object):
    def __init__(self, minimum: torch.tensor, maximum: torch.tensor):
        self.min = minimum
        self.max = maximum

    def __call__(self, x: torch.tensor):
        return (x-self.min)/(self.max - self.min)


def draw_scatter_fn(y, y_pred):
    fig, axs = plt.subplots(1, 1)
    axs.scatter(y, y_pred)
    axs.set_ylabel("Predicted Value", fontsize=10)
    axs.set_xlabel("True Value", fontsize=10)
    return fig


def log_train(log_writer, loss, mae, mse, rmse, ccc, pcc, symptoms, iter_idx):
    log_writer.add_scalar('Loss/Train', loss, iter_idx)
    log_writer.add_scalar('MAE/Train', mae, iter_idx)
    log_writer.add_scalar('MSE/Train', mse, iter_idx)
    log_writer.add_scalar('RMSE/Train', rmse, iter_idx)
    for sym, c, p in zip(symptoms, ccc, pcc):
        log_writer.add_scalar('{}-CCC/Train'.format(sym), c, iter_idx)
        log_writer.add_scalar('{}-PCC/Train'.format(sym), p, iter_idx)


def log_val(log_writer, vmae, vmse, vrmse, vccc, vpcc, symptoms, iter_idx):
    for mae, mse, rmse, ccc, pcc, sym in zip(vmae, vmse, vrmse, vccc, vpcc, symptoms):
        log_writer.add_scalar('{}-MAE/Validation'.format(sym), mae, iter_idx)
        log_writer.add_scalar('{}-MSE/Validation'.format(sym), mse, iter_idx)
        log_writer.add_scalar('{}-RMSE/Validation'.format(sym), rmse, iter_idx)
        log_writer.add_scalar('{}-PCC/Validation'.format(sym), pcc, iter_idx)
        log_writer.add_scalar('{}-CCC/Validation'.format(sym), ccc, iter_idx)

