import sys
import torch
from torch import functional as F
import torch.optim as optim

from architecture import *
from utils import *


def get_lfb(cnf, backbone, neck, inputs):
    if inputs.shape[1] < cnf.input_shape[1] * (cnf.window * 2 + 1):
        pad = cnf.input_shape[1] * (cnf.window * 2 + 1) - inputs.shape[1]
        inputs = F.pad(inputs, (0, 0, 0, 0, 0, 0, 0, pad), "constant")
    inputs = torch.split(inputs, cnf.input_shape[1], dim=1)
    inputs = torch.stack(inputs)
    inputs = inputs.permute(1, 0, 2, 3, 4, 5)
    b, lf, t, c, h, w = inputs.shape
    lfb_feats = inputs.reshape(b * lf, t, c, h, w).clone().detach()
    inputs = inputs[:, cnf.window, ...].squeeze(1).detach()
    with torch.no_grad() and autocast():
        frames = backbone(lfb_feats)
        lfb_feats = neck(frames)
        lfb_feats = lfb_feats.detach()
        lfb_feats = lfb_feats.reshape(b, lf, -1)
    return inputs, lfb_feats


def get_lfb_val(cnf, backbone, neck, inputs):
    if (inputs.shape[1] % cnf.input_shape[1]) != 0:
        remainder = -(inputs.shape[1] // -cnf.input_shape[1])  # ceiling division
        pad = cnf.input_shape[1] * remainder - inputs.shape[1]
        inputs = F.pad(inputs, (0, 0, 0, 0, 0, 0, 0, pad), "constant")
    inputs = torch.split(inputs, cnf.input_shape[1], dim=1)
    inputs = torch.stack(inputs)
    inputs = inputs.permute(1, 0, 2, 3, 4, 5)
    b, cl, t, c, h, w = inputs.shape
    lfb_clips = inputs.reshape(b * cl, t, c, h, w).clone().detach()
    lfb_feats = torch.zeros(b * cl, backbone.module.interim_dim)
    with torch.no_grad() and autocast():
        for i in range(lfb_clips.shape[0]):
            frames = backbone(lfb_clips[i].unsqueeze(0))
            feats = neck(frames)
            feats = feats.detach()
            lfb_feats[i, :] = feats[:]
    # lfb_feats = lfb_feats.reshape(b, cl, -1)
    lfb = list()
    for i in range(-cnf.window, cnf.window+1):
        lfb.append(lfb_feats.roll(i, 0))
    lfb = torch.stack(lfb)
    lfb = lfb.permute(1, 0, 2)
    return inputs.reshape(b * cl, t, c, h, w), lfb


def train_log(cnf, epoch, net, train_loader, log_writer, optimizer, scaler):
    num_iter = len(train_loader)
    net = [el.train() for el in net]
    backbone, neck, head = net
    for batch_idx, (inputs, labels, _) in enumerate(train_loader):
        iter_idx = (epoch * num_iter) + batch_idx
        inputs, labels = inputs.cuda(), labels.cuda()
        if cnf.window > 0:
            inputs, lfb_feats = get_lfb(cnf, backbone, neck, inputs)
        else:
            lfb_feats = None
        optimizer.zero_grad()
        with autocast():
            frames = backbone(inputs)
            clips = neck(frames)
            outputs = head(clips, lfb_feats)
            y_out = outputs['pred']

            mae = F.l1_loss(y_out, labels, reduction='mean')
            mse = F.mse_loss(y_out, labels, reduction='mean')
            rmse = torch.sqrt(mse)

            pcc = PCC(y_out, labels)
            ccc = CCC(y_out, labels)
            ccc_loss = (1 - pcc).mean()
            loss = rmse #+ ccc_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        log_train(log_writer, loss, mae, mse, rmse, ccc, pcc, cnf.symptom_names, iter_idx)

        sys.stdout.write(
            '\r {}| Epoch [{}/{}] Iter[{}/{}]\t loss: {:.2f} \t MAE: {:.2f} \t MSE: {:.2f} \t RMSE: {:.2f} \t CCC:{} \t PCC:{} '.format(
                cnf.dataset_name,
                epoch,
                cnf.num_epochs,
                batch_idx + 1,
                num_iter,
                loss.item(),
                mae.item(),
                mse.item(),
                rmse.item(),
                ['%.2f' % elem for elem in ccc.tolist()],
                ['%.2f' % elem for elem in pcc.tolist()]
            )
        )
        sys.stdout.flush()
        torch.cuda.empty_cache()


def val_log(cnf, epoch, net, test_loader, log_writer, last_loss):
    net = [el.eval() for el in net]
    y_pred, y_true = val_loop(cnf, test_loader, net)

    val_mae, val_mse, val_rmse, val_pcc, val_ccc = eval_metrics(y_pred, y_true)
    loss = (1 - val_ccc).mean()
    sys.stdout.write('\n {}| Validation [{}/{}] \t loss:{:.2f} \t MAE: {} \t RMSE: {} \t PCC: {} \t CCC: {}'.format(
        cnf.dataset_name,
        epoch,
        cnf.num_epochs,
        loss.item(),
        ['%.2f' % elem for elem in val_mae.tolist()],
        ['%.2f' % elem for elem in val_rmse.tolist()],
        ['%.2f' % elem for elem in val_pcc.tolist()],
        ['%.2f' % elem for elem in val_ccc.tolist()]
    ))
    log_val(log_writer, val_mae, val_mse, val_rmse, val_ccc, val_pcc, cnf.symptom_names, epoch)
    for sym_idx, sym in enumerate(cnf.symptom_names):
        scatter = draw_scatter_fn(y_true[:, sym_idx], y_pred[:, sym_idx])
        log_writer.add_figure('Pred vs Actual: {}'.format(sym), scatter, epoch)
    log_writer.flush()
    sys.stdout.write('\n')
    torch.cuda.empty_cache()
    last_loss.append(loss)


@torch.no_grad()
def val_loop(cnf, test_loader, net):
    # assumes batch_size=1 in validation
    backbone, neck, head = net
    y_true = torch.zeros(len(test_loader.dataset), len(cnf.symptoms))
    y_pred = torch.zeros(len(test_loader.dataset), len(cnf.symptoms))
    for batch_idx, (inputs, labels, video_idx) in enumerate(test_loader):
        # inputs, labels = inputs.cuda(), labels.cuda()
        # input shape (b*n_clips, t, ch, w, h)
        # lfb shape (b*n_clips, window*2+1, interim_size)
        if cnf.window > 0:
            inputs, lfb_feats = get_lfb_val(cnf, backbone, neck, inputs)
        else:
            lfb_feats = None
            inputs = torch.split(inputs, cnf.input_shape[1], dim=1)
            inputs = torch.stack(inputs[:-1]) if len(inputs) > 1 else torch.stack(inputs)
            inputs = inputs.squeeze(1)
        vid_pred = torch.zeros(len(inputs), len(cnf.symptoms))
        for clip_idx, clip in enumerate(inputs):
            with autocast():
                frame_feat = backbone(clip.unsqueeze(0).cuda())
                clip_feat = neck(frame_feat)
                if lfb_feats is not None:
                    lfb = lfb_feats[clip_idx].unsqueeze(0).cuda()
                else:
                    lfb = None
                outputs = head(clip_feat, lfb)
                vid_pred[clip_idx] = outputs['pred']
        start_idx = batch_idx * 1
        end_idx = start_idx + 1
        end_idx = end_idx if end_idx <= len(y_true) else len(y_true)
        y_pred[start_idx: end_idx] = vid_pred.mean(0)
        y_true[start_idx: end_idx] = labels[:]
    return y_pred, y_true


def eval_metrics(y_hat, y):
    mae = F.l1_loss(y_hat, y, reduction='none').mean(0)
    mse = F.mse_loss(y_hat, y, reduction='none').mean(0)
    rmse = torch.sqrt(mse)
    pcc = PCC(y_hat, y)
    ccc = CCC(y_hat, y)
    return mae, mse, rmse, pcc, ccc


def get_optimizer(net, cnf, opt: str = 'Adam'):
    backbone, neck, head = net
    if opt == 'Adam':
        optimizer = optim.Adam(
            [{'params': backbone.parameters()}, {'params': neck.parameters()}, {'params': head.parameters()}],
            lr=cnf.lr,
            weight_decay=5e-3
        )
    elif opt == 'SGD':
        optimizer = optim.SGD(
            [{'params': backbone.parameters()}, {'params': neck.parameters()}, {'params': head.parameters()}],
            lr=cnf.lr,
            weight_decay=5e-4
        )
    else:
        raise NotImplementedError
    return optimizer
