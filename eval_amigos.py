from datetime import datetime

import matplotlib.pyplot as plt
from torch.utils import tensorboard
from torch.utils import data

from config import *
from train_utils import *
from AMIGOS import AMIGOS
from AMIGOS import amigos_utils

torch.backends.cudnn.enabled = False

kwargs = {'num_workers': 4, 'pin_memory': True, 'shuffle': True}
cnf = get_config(sys.argv)


def main(model_lst):
    space_transform = transforms.Compose([
        transforms.Resize((cnf.input_shape[0], cnf.input_shape[0])),
        transforms.ToTensor()
    ])
    target_transform = [torch.FloatTensor, amigos_utils.custom_transforms.AnnotatorsAverage()]
    target_transform = transforms.Compose(target_transform)
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
    if cnf.lfb:
        backbone = nn.DataParallel(Backbone()).cuda()
        neck = nn.DataParallel(Neck(backbone.module.interim_dim)).cuda()
        head = nn.DataParallel(NLB_Head_AV(backbone.module.interim_dim)).cuda()
    else:
        backbone = nn.DataParallel(Backbone()).cuda()
        neck = nn.DataParallel(Neck(backbone.module.interim_dim)).cuda()
        head = nn.DataParallel(LN_Head_VA(backbone.module.interim_dim, cnf.variance)).cuda()
    y_hat, y = list(), list()
    for model in model_lst:
        fname = os.path.basename(model).split('.')[0]
        idx = fname.replace('pid_', '')
        val_idx = amigos_utils.get_indices_in_set(val_dataset.data, [idx])
        val_set = data.Subset(val_dataset, val_idx)
        test_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=1,
            collate_fn=amigos_utils.series_collate,
            **kwargs
        )
        print('AMIGO {}: Test samples: {}'.format(idx, len(test_loader)))
        if len(test_loader) == 0:
            continue
        state_dicts = torch.load(model)
        backbone.load_state_dict(state_dicts['backbone'])
        neck.load_state_dict(state_dicts['neck'])
        head.load_state_dict(state_dicts['head'])
        net = (backbone, neck, head)
        net = (el.eval() for el in net)
        y_pred, y_true = val_loop(cnf, test_loader, net)
        y_hat.append(y_pred.cpu())
        y.append(y_true.cpu())
        torch.cuda.empty_cache()

    y_hat = torch.cat(y_hat)
    y = torch.cat(y)
    mae, mse, rmse, pcc, ccc = eval_metrics(y_hat, y)
    print(mae, rmse, pcc, ccc)
    log_writer = tensorboard.SummaryWriter(
        log_dir=os.path.join(*[cnf.log_dir, now + cnf.model_basename + '_eval'])
    )
    log_val(log_writer, mae, mse, rmse, ccc, pcc, cnf.symptom_names, 0)

    for sym_idx, sym in enumerate(cnf.symptom_names):
        scatter = draw_scatter_fn(y[:, sym_idx], y_hat[:, sym_idx])
        log_writer.add_figure('Pred vs Actual: {}'.format(sym), scatter, 0)
    log_writer.flush()


if __name__ == "__main__":
    cnf = get_config(sys.argv)
    cnf.input_shape = [224, 16]
    # cnf.lfb = False
    path = 'models/{}/'.format(cnf.model_basename)
    now = datetime.now().strftime('%b%d_%H-%M-%S_')
    weight_files = os.listdir(path)
    weight_files = [os.path.join(path, fname) for fname in weight_files]
    print('amigos: {}'.format(len(weight_files)))
    main(weight_files)
