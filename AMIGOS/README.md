# AMIGOS
Pytorch DataLoader for face video data of the AMIGOS dataset [1].

## Requirements
- Python 3.8
- ffmpeg version n4.3.1
- python packages in requirements.txt

## Preprocess
Unzip files: `python AMIGOS/preprocess/unzipfiles.py <folder_path> <destination_folder_path>`

Transform External_Annotations.xlsx to json: `python AMIGOS/preprocess/External_Annotations_xlsxtojson.py <file_path>`

Split videos to frames: `python AMIGOS/preprocess/mov_split_frames.py <origin_folder_path> <destination_folder_path>`

## Usage
```python
from torch.utils import data
from torchvision import transforms
from AMIGOS import AMIGOS


if __name__ == '__main__':
    amigos = AMIGOS(
        root_path='Frames',
        annotation_path='amigos.json',
        spatial_transform=transforms.ToTensor(),
        feature_type='RGB'
    )
    data_loader = data.DataLoader(amigos)
    for i, data in enumerate(data_loader):
        clip = data[0]
        labels = data[1]
```

Expected folder structure:
```
Root dir
└── Unzipped Folder
    ├── Video Folder 1
    │   ├── 1 # segment
    │   │   └── image_00001.jpg # frame
    │   ├── 2
    │   │   └── image_00001.jpg
    │   └── 3
    │       └── image_00001.jpg
    └── Video Folder 1
        ├── 1
        │   └── image_00001.jpg
        ├── 2
        │   └── image_00001.jpg
        ├── 3
        │   └── image_00001.jpg
        └── 4
            └── image_00001.jpg

```
## Feature Extraction
To use features extracted from the RGB images, the folder structure needs to be the same as for the RGB and contain a single json file with features for the entire segment clip:

```json
{
  "features": {
    "1": {"feat1": 0, "feat2": 0, "feat3": 0},
     "2": {"feat1": 0, "feat2": 0, "feat3": 0}
  },
  "Other Metadata": {}
}
```

## TODO
- [x] Handle features in json files
- [x] Method to get user_id for leave one out training
## Bib
[1] J. A. Miranda Correa, M. K. Abadi, N. Sebe, and I. Patras, ‘AMIGOS: A Dataset for Affect, Personality and Mood Research on Individuals and Groups’, IEEE Trans. Affective Comput., pp. 1–1, 2018, doi: 10.1109/TAFFC.2018.2884461.
