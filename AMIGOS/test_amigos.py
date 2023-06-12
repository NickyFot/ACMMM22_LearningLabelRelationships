from torchvision import transforms
import torch
from amigos_dataset import AMIGOS
from utils import custom_transforms


if __name__ == '__main__':
    amigos = AMIGOS(
        root_path='AU',
        annotation_path='amigos.json',
        spatial_transform=transforms.Compose([custom_transforms.ColumnSelect(keys=['smiling']), torch.FloatTensor]),
        target_transform=transforms.Compose([torch.FloatTensor, custom_transforms.AnnotatorsAverage()]),
        feature_type='Meta'
    )
    print(len(amigos))
    sample = amigos[0]
    print(sample[0].shape, sample[1].shape)
