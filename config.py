import os
import argparse

import torch
from NESS import col_index


def get_config(sysv):
    parser = argparse.ArgumentParser(description='Training variables.')
    parser.add_argument('--model_basename', default='baseline')
    parser.add_argument('--log_dir', default='logs')
    parser.add_argument('--body', action='store_true', default=False)
    parser.add_argument('--variance', action='store_true', default=False)
    parser.add_argument('--lfb', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=torch.cuda.device_count())
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--window', type=int, default=2)
    parser.add_argument('--dataset_root', default='/import/nobackup_mmv_ioannisp/shared/datasets/NESS/dataset')
    parser.add_argument('--dataset_name', default='NESS')
    parser.add_argument('--dataset_subsection', default='faces')
    parser.add_argument('--scale', default='panss')
    parser.add_argument('--label_path', default='panss_labels.json')
    parser.add_argument('--symptom_names', nargs='+', default=['N3: Poor Rapport',  'N6: Lack of Spontaneity', 'N1: Blunted Affect', 'total'])
    parser.add_argument('--input_shape', nargs='+', default=[224, 32])
    parser.add_argument('--downsample', type=int, default=8)
    parser.add_argument('--pretrained', help='dir of pretrained files')
    parser.add_argument('--lamda', type=float, default=1.0)

    parser.add_argument('amigos', nargs='+', default=[])
    parser.add_argument('amigos_ignore', nargs='+', default=[])

    #contrastive
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--n_views', type=int, default=2)
    args, _ = parser.parse_known_args(sysv)
    if args.dataset_name == 'NESS':
        args.symptoms = col_index(args.symptom_names, args.scale)
    else:
        args.symptoms = list(range(len(args.symptom_names)))
    # now = datetime.now().strftime('%b%d_%H-%M-%S_')
    # args.log_dir = os.path.join('logs', now + args.model_basename)

    # args.lfb = True

    return args
