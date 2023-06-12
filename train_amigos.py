import os
from datetime import datetime

import random

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils import tensorboard
from torch.utils import data
from torch.cuda.amp import GradScaler

from torchvision import transforms

from config import *
from train_utils import *
from AMIGOS import AMIGOS
from AMIGOS import amigos_utils

torch.backends.cudnn.enabled = False

kwargs = {'num_workers': 4, 'pin_memory': True, 'shuffle': True}
cnf = get_config(sys.argv)


def main() -> None:
    log_writer = tensorboard.SummaryWriter(
        log_dir=os.path.join(*[cnf.log_dir, 'AMIGOS', now + cnf.model_basename])
    )
    load_transform = transforms.Compose([
        IgnoreFiles('face-0'),
        TemporalDownSample(cnf.downsample),
        RandomSequence(cnf.input_shape[1] * (cnf.window * 2 + 1))
    ])
    temporal_transform = transforms.Compose([
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.4168, 0.3074, 0.2607], [0.2426, 0.1997, 0.1870])
    ])
    space_transform = transforms.Compose([
        transforms.Resize((cnf.input_shape[0], cnf.input_shape[0])),
        transforms.ToTensor()
    ])

    target_transform = [torch.FloatTensor, amigos_utils.custom_transforms.AnnotatorsAverage()]
    target_transform = transforms.Compose(target_transform)
    dataset = AMIGOS(
        root_path=cnf.dataset_root,
        annotation_file=cnf.label_path,
        spatial_transform=space_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        load_transform=load_transform
    )
    temporal_transform = transforms.Compose([
        transforms.Normalize([0.4168, 0.3074, 0.2607], [0.2426, 0.1997, 0.1870])
    ])
    load_transform = transforms.Compose([
        IgnoreFiles('face-0'),
        TemporalDownSample(cnf.downsample),
        # RandomSequence(cnf.input_shape[1] * (cnf.window * 2 + 1))
    ])
    val_dataset = AMIGOS(
        root_path=cnf.dataset_root,
        annotation_file=cnf.label_path,
        spatial_transform=space_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        load_transform=load_transform
    )
    amigos_idx = amigos_utils.get_subject_idx(dataset.data)
    if cnf.amigos_ignore:
        print(amigos_idx)
        for ign in cnf.amigos_ignore:
            if ign in amigos_idx:
                amigos_idx.remove(ign)
    for amigo in amigos_idx:
        val_idx = amigos_utils.get_indices_in_set(val_dataset.data, [amigo])
        train_idx = amigos_utils.get_indices_in_set(
            dataset.data,
            [amig_idx for amig_idx in amigos_utils.get_subject_idx(dataset.data) if amig_idx != amigo]
        )

        assert len([value for value in train_idx if value in val_idx]) == 0

        new_train = random.sample(train_idx, len(train_idx)//5)
        print("AMIGO {} - Test: {} Train: {}".format(amigo, len(val_idx), len(new_train)))

        val_set = data.Subset(val_dataset, val_idx)
        train_dataset = data.Subset(dataset, new_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cnf.batch_size,
            collate_fn=amigos_utils.series_collate,
            **kwargs
        )
        test_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=1,
            collate_fn=amigos_utils.series_collate,
            **kwargs
        )
        if cnf.lfb:
            backbone = nn.DataParallel(Backbone()).cuda()
            neck = nn.DataParallel(Neck(backbone.module.interim_dim)).cuda()
            head = nn.DataParallel(NLB_Head_AV(backbone.module.interim_dim)).cuda()
        else:
            backbone = nn.DataParallel(Backbone()).cuda()
            neck = nn.DataParallel(Neck(backbone.module.interim_dim)).cuda()
            head = nn.DataParallel(LN_Head_VA(backbone.module.interim_dim)).cuda()
        net = [backbone, neck, head]
        optimizer = optim.SGD(
            [{'params': backbone.parameters()}, {'params': neck.parameters()}, {'params': head.parameters()}],
            lr=cnf.lr,
            weight_decay=5e-4
        )
        scaler = GradScaler()
        last_loss = list()
        lr = cnf.lr
        for epoch in range(cnf.num_epochs):
            if (epoch % 5 == 0) and (epoch != 0):
                lr *= 0.1
                optimizer.param_groups[0]['lr'] = lr / 2.6
                optimizer.param_groups[1]['lr'] = lr
                optimizer.param_groups[2]['lr'] = lr
            num_iter = len(train_loader)
            net = [el.train() for el in net]
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
                    distance, similarity = info_recon_loss(clips, labels)
                    cnt_mse = F.mse_loss(distance, similarity, reduction='mean')
                    cnt_rmse = torch.sqrt(cnt_mse)
                    loss = (1-ccc).mean() + cnf.lamda * cnt_rmse

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
                del inputs
                del lfb_feats
                del labels  # free mem
                torch.cuda.empty_cache()
                # break
            val_log(cnf, epoch, net, test_loader, log_writer, last_loss)
            if last_loss[-1] == min(last_loss):
                savemodel = 'models/{}'.format(cnf.model_basename)
                if not os.path.exists(savemodel):
                    os.makedirs(savemodel)
                torch.save({
                    'epoch': epoch,
                    'backbone': backbone.state_dict(),
                    'neck': neck.state_dict(),
                    'head': head.state_dict()
                },
                    os.path.join(savemodel, 'pid_{}.pth.tar'.format(amigo))
                )


if __name__ == '__main__':
    now = datetime.now().strftime('%b%d_%H-%M-%S_')
    cnf.input_shape = [224, 16]
    main()
