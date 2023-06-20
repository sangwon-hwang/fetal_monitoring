import torch
from torch import nn, einsum
import torch.nn.functional as F

from torch.optim import Adam
from torchvision import transforms as T, utils

from pathlib import Path
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

from PIL import Image
from pathlib import Path
from functools import partial
from denoising_diffusion_pytorch import *
from torch.utils.data import (Dataset, 
                              DataLoader)
from torchvision import transforms as T, utils


class custom_dataset(Dataset):
    def __init__(self,
                 folder,
                 image_size,
                 ext,
                 augment_horizontal_flip = False,
                 convert_image_to = None):
        super().__init__()
        assert ext in ['jpg', 'jpeg', 'png', 'tiff'], \
            f'extension should be one of jpg jpeg png tiff'

        self.folder = folder
        self.image_size = image_size
        self.paths = [p for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) \
            if exists(convert_image_to) else nn.Identity()
        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        # if not img.model=='RGB':
        #     img = img.convert('RGB')
        
        return self.transform(img)


if __name__ == "__main__":
    folder = '/home/sangwon/sources/diffusers/data/embryo_image/train'
    image_size = 256
    train_batch_size = 16
    ext = 'png'
    augment_horizontal_flip = False
    convert_image_to = None

    ds = custom_dataset(folder, 
                        image_size, 
                        ext,
                        augment_horizontal_flip, 
                        convert_image_to)
    print(ds[0].shape)

