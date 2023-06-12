from torch.utils import data
from torch.nn.utils import rnn

from .utils import *


class AMIGOS(data.Dataset):
    """Class to handle AMIGOS Dataset.

    A class to handle and get video data from the AMIGOS dataset.
        Typical usage example:
        amigos = AMIGOS(
        root_path=<path_to_dataset>,
        annotation_path='amigos.json',
        spatial_transform=transforms.ToTensor(),
        feature_type='RGB'
    )

    """
    def __init__(
            self,
            root_path: str,
            annotation_file: str,
            spatial_transform: callable = None,
            temporal_transform: callable = None,
            target_transform: callable = None,
            load_transform: callable = None,
            lfb: bool = False,
            contrast: bool = False,
            n_views: int = 2
    ):
        """
        Dataset constructor
        :param root_path: (str) path to content root
        :param annotation_path: (str) path to annotations file
        :param spatial_transform: (callable) transformation to apply to each frame
        :param temporal_transform: (callable) transformation to apply to clip
        :param target_transform: (callable) transformation to apply to target
        :param feature_type: (str) feature type to return Options: Meta, RGB
        :returns AMIGOS Dataset object
        """
        self.root_path = root_path
        self.annotation_path = annotation_file
        self.feature_type = 'Faces'
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.load_transform = load_transform
        self.loader = functools.partial(video_loader, image_loader=pil_loader)
        self.data = self._make_dataset(
            root_path,
            annotation_file,
            file_extension='.jpg'
        )
        self.lfb = lfb  # TODO: before and after clips in __getitem__
        self.contrast = contrast
        self.n_views = n_views
        self.labels = [x['label'] for x in self.data]
        self.indices = list(range(0, len(self.data)))

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self.labels = [x['label'] for x in self.data]
        self.indices = list(range(0, len(self.data)))

    def __getaugmented__(self, index):
        path = self.data[index]['video']
        clips = list()
        target = self.data[index]['label']
        if self.target_transform is not None:
            target = self.target_transform(target)
        for i in range(self.n_views):
            if self.load_transform:
                clip = self.loader(path, time_transform=self.load_transform)
            else:
                clip = self.loader(path)
            if self.spatial_transform is not None:
                clip = [self.spatial_transform(img) for img in clip if img]
            clip = torch.stack(clip, 0)
            if self.temporal_transform is not None:
                clip = self.temporal_transform(clip)
            clips.append(clip)
        clips = rnn.pad_sequence(clips, batch_first=True)
        return clips, target, self.data[index]['video_id']

    def __getclip__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (clip, target, video_idx)
        """
        # TODO: handle lfb idx here
        path = self.data[index]['video']
        if self.load_transform:
            clip = self.loader(path, time_transform=self.load_transform)
        else:
            clip = self.loader(path)
        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip if img]
        clip = torch.stack(clip, 0)
        if self.temporal_transform is not None:
            clip = self.temporal_transform(clip)
        target = self.data[index]['label']
        if self.target_transform is not None:
            target = self.target_transform(target)
        return clip, target, self.data[index]['video_id']

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (clip, target, video_idx)
        """
        # TODO: handle lfb idx here
        if self.contrast:
            return self.__getaugmented__(index)
        else:
            return self.__getclip__(index)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _make_dataset(
            root_path: str,
            annotation_path: str,
            file_extension: str = '.json'
    ) -> list:
        dataset = load_annotation_data(os.path.join(root_path, annotation_path))
        video_names, annotations = get_video_names_and_annotations(dataset)
        filenames = get_file_names(root_path, file_extension)
        if file_extension == '.jpg':
            filenames = list(set([os.path.dirname(x) for x in filenames]))
        dataset = []
        for idx in range(len(filenames)):
            froot = filenames[idx]
            segment = os.path.basename(froot)
            subject_vid = os.path.basename(os.path.dirname(froot)).split('_')[:2]
            subject_vid.append(segment)
            key = '_'.join(subject_vid)
            video_path = filenames[idx]

            if key not in video_names:
                continue
            j = video_names.index(key)
            if not os.path.exists(video_path):
                continue
            if len(annotations[j]) == 0:
                continue
            sample = {
                'video': video_path,
                'video_id': video_names[j],
                'label': annotations[j]
            }
            dataset.append(sample)
        return dataset
