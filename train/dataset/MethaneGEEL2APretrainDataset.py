import logging
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy.ndimage
import tifffile
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
# Band order: ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12')
s2_mean = np.array(
    [855.06971511, 
     1071.58087384, 
     1486.44923875, 
     2025.45348097, 
     2357.59363062,
     2644.80530558, 
     2816.5734476,  
     2925.41640669, 
     3030.1418974,  
     3038.04763191,
     3895.84487789, 
     3332.52674011
    ],
    dtype=np.float32,
)
s2_std = np.array(
    [567.3335958, 
     628.26232062, 
     692.82233274, 
     844.46846068, 
     848.82207617,
     797.86434453, 
     798.79855744, 
     804.58738457, 
     784.93118895, 
     771.53437732,
     843.95042025, 
     901.15392137
    ],
    dtype=np.float32,
)

class RandomCrop:
    def __init__(self, crop_size):
        self.crop_height, self.crop_width = crop_size
        self.center_size = 30

    def __call__(self, sample):
        channels, height, width = sample['image'].shape
        if random.random() < 0.5:
            center_x = width // 2
            center_y = height // 2
            center_left = center_x - self.center_size // 2
            center_right = center_x + self.center_size // 2
            center_top = center_y - self.center_size // 2
            center_bottom = center_y + self.center_size // 2

            left = random.randint(0, max(0, center_left - self.crop_width))
            top = random.randint(0, max(0, center_top - self.crop_height))

            if left + self.crop_width < center_right:
                left = center_right - self.crop_width
            if top + self.crop_height < center_bottom:
                top = center_bottom - self.crop_height
        else:
            top = random.randint(0, height - self.crop_height)
            left = random.randint(0, width - self.crop_width)

        out = {
            'image': sample['image'][:, top:top + self.crop_height, left:left + self.crop_width],
            'label': sample['label'][top:top + self.crop_height, left:left + self.crop_width]
        }
        
        return out

class RandomCenterCrop:
    def __init__(self, crop_size, center_size=30):
        self.crop_height, self.crop_width = crop_size
        self.center_size = center_size

    def __call__(self, sample):
        if isinstance(sample, dict):
            image = sample["image"]
            label = sample.get("label")
        else:
            image = sample
            label = None

        _, height, width = image.shape
        center_x = width // 2
        center_y = height // 2
        center_left = center_x - self.center_size // 2
        center_right = center_x + self.center_size // 2
        center_top = center_y - self.center_size // 2
        center_bottom = center_y + self.center_size // 2

        left = random.randint(0, max(0, center_left - self.crop_width))
        top = random.randint(0, max(0, center_top - self.crop_height))

        if left + self.crop_width < center_right:
            left = center_right - self.crop_width
        if top + self.crop_height < center_bottom:
            top = center_bottom - self.crop_height

        cropped_image = image[:, top : top + self.crop_height, left : left + self.crop_width]

        if isinstance(sample, dict):
            sample["image"] = cropped_image
            if label is not None:
                sample["label"] = label[top : top + self.crop_height, left : left + self.crop_width]
            else:
                sample.pop("label", None)
            return sample

        return cropped_image

class CenterCrop:
    def __init__(self, crop_size, center_size=20):
        self.crop_height, self.crop_width = crop_size
        self.center_size = center_size

    def __call__(self, sample):
        channels, height, width = sample['image'].shape
        center_x = width // 2
        center_y = height // 2

        left = center_x - self.crop_width
        top = center_y - self.crop_height

        out = {
            'image': sample['image'][:, top:top + self.crop_height, left:left + self.crop_width],
            'label': sample['label'][top:top + self.crop_height, left:left + self.crop_width]
        }
        
        return out

class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, sample):
        angle = random.choice(self.angles)
        out = {
            'image': np.array([scipy.ndimage.rotate(channel, angle, reshape=False) for channel in sample['image']]),
            'label': scipy.ndimage.rotate(sample['label'], angle, reshape=False, order=0, cval=0)
        }
        
        return out
    
class RandomFlip:
    def __init__(self, horizontal_prob=0.5, vertical_prob=0.5):
        self.horizontal_prob = horizontal_prob
        self.vertical_prob = vertical_prob

    def __call__(self, sample):
        image_array = sample['image']
        if random.random() < self.horizontal_prob:
            image_array = np.flip(image_array, axis=2)
        
        # 随机垂直翻转
        if random.random() < self.vertical_prob:
            image_array = np.flip(image_array, axis=1)
        
        sample['image'] = image_array
        return sample
    
class ToTensor:
    def __call__(self, sample):
        out = {'image': torch.from_numpy(sample['image'].copy()), 'label': sample['label']}
        return out
    

class MocoTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        sample1 = self.transforms(sample)
        sample2 = self.transforms(sample)
        sample1['image2'] = sample2['image']
        return sample1


class NormalizeLast:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-6)


class RSBYOLTransform:
    def __init__(self, size: int = 32, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.size = size
        if mean is None or std is None:
            raise ValueError("RSBYOLTransform requires mean and std for normalization.")
        self.norm = NormalizeLast(mean, std)

    def _rand_resized_crop(self, x: np.ndarray, scale=(0.6, 1.0)) -> np.ndarray:
        h, w = x.shape[-2:]
        s = random.uniform(*scale)
        new = max(1, int(self.size * s))
        new = min(new, min(h, w))
        i = random.randint(0, max(0, h - new))
        j = random.randint(0, max(0, w - new))
        x = x[:, i:i + new, j:j + new]
        zoom = self.size / new
        x = np.stack([scipy.ndimage.zoom(c, zoom, order=1) for c in x], axis=0)
        return x

    def _gauss_blur(self, x: np.ndarray, sigma=(0.1, 1.0)) -> np.ndarray:
        s = random.uniform(*sigma)
        return np.stack([scipy.ndimage.gaussian_filter(c, s) for c in x], axis=0)

    def _channel_jitter(self, x: np.ndarray, add=0.05, mul=0.05) -> np.ndarray:
        noise_mul = np.random.randn(x.shape[0], 1, 1).astype(np.float32) * mul
        x = x * (1.0 + noise_mul)
        x = x + np.random.randn(*x.shape).astype(np.float32) * add
        return x

    def _channel_dropout(self, x: np.ndarray, p=0.1) -> np.ndarray:
        mask = (np.random.rand(x.shape[0]) > p).astype(np.float32)
        return x * mask[:, None, None]

    def _erase(self, x: np.ndarray, s=(0.1, 0.3)) -> np.ndarray:
        sz = random.randint(max(1, int(self.size * s[0])), max(1, int(self.size * s[1])))
        sz = min(sz, self.size)
        i = random.randint(0, max(0, self.size - sz))
        j = random.randint(0, max(0, self.size - sz))
        erase_patch = np.random.randn(x.shape[0], sz, sz).astype(np.float32) * 0.1
        x[:, i:i + sz, j:j + sz] = erase_patch
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # print("RSBYOL")
        x = np.asarray(x, dtype=np.float32, order="C")
        if random.random() < 0.5:
            x = np.flip(x, axis=2)
        if random.random() < 0.5:
            x = np.flip(x, axis=1)
        if random.random() < 0.5:
            x = np.rot90(x, k=random.choice([1, 2, 3]), axes=(1, 2))
        x = self._rand_resized_crop(x, scale=(0.6, 1.0))
        if random.random() < 0.8:
            x = self._gauss_blur(x)
        if random.random() < 0.8:
            x = self._channel_jitter(x)
        if random.random() < 0.3:
            x = self._channel_dropout(x)
        if random.random() < 0.5:
            x = self._erase(x)
        x = self.norm(x)
        return x.copy()


def _resolve_files(
    data_dir: Path,
    extensions: Sequence[str],
) -> list[Path]:
    files: list[Path] = []
    for ext in extensions:
        files.extend(sorted(data_dir.rglob(f"*{ext}")))
    return files


class MethaneGEEL2APretrainDataset(Dataset):
    def __init__(
        self,
        transform,
        data_dir,
        channels: Sequence[int] | None = None,
        file_extensions: Sequence[str] | None = None,
    ):
        self.transform = transform
        self.data_dir = Path(data_dir)

        self.file_extensions = file_extensions or (".tif", ".tiff")
        self.files = _resolve_files(self.data_dir, self.file_extensions)

        logger.info("Loaded %d chips from %s", len(self.files), self.data_dir)

        default_channels = list(range(len(s2_mean)))
        self.channels = list(channels) if channels is not None else default_channels
        self.channels = self._validate_channels(self.channels)

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _validate_channels(channels: Iterable[int]) -> list[int]:
        channel_list = sorted(set(int(c) for c in channels))
        if not channel_list:
            raise ValueError("Channels must contain at least one band index.")
        for c in channel_list:
            if c < 0:
                raise ValueError(f"Channel indices must be non-negative. Got {c}.")
            if c >= len(s2_mean):
                logger.warning(
                    "Channel index %d exceeds current mean/std placeholder length (%d). "
                    "Ensure `s2_mean` and `s2_std` are updated accordingly.",
                    c,
                    len(s2_mean),
                )
        return channel_list

    @staticmethod
    def open_image(img_path: Path) -> np.ndarray:
        img = tifffile.imread(str(img_path))
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        elif img.ndim == 3 and img.shape[0] != len(s2_mean) and img.shape[-1] == len(s2_mean):
            img = np.moveaxis(img, -1, 0)
        return img.astype(np.float32, copy=False)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = self.open_image(img_path)
        if image.shape[0] <= max(self.channels):
            raise ValueError(
                f"Image at {img_path} only has {image.shape[0]} bands, "
                f"but channels {self.channels} were requested."
            )
        image = image[self.channels, :, :]

        view1 = self.transform(image)
        view2 = self.transform(image)
        return view1, view2
    
class MethaneGEEL2AMAEPretrainDataset(Dataset):
    def __init__(self, transform, data_path, channels, crop_size = None):
        
        self.df = pd.read_csv(data_path)
        # self.df = self.df[self.df['cloud_covered'] == 0]
        print(f'total samples {len(self.df)}')
        self.transform = transform
        self.channels = channels
        self.epsilon = 1e-9
        self.crop_size = crop_size

    def __len__(self):
        return len(self.df)
    
    def open_image(self, img_path):
        img = tifffile.imread(img_path)
        return img.astype(np.float32)
    
    def random_crop(self, x, crop_size=(32, 32)):
        h, w = x.shape[1], x.shape[2]
        crop_h, crop_w = crop_size
        if h > crop_h and w > crop_w:
            x_start = random.randint(0, h - crop_h)
            y_start = random.randint(0, w - crop_w)
            x = x[:, x_start:x_start + crop_h, y_start:y_start + crop_w]
        return x

    def __getitem__(self, idx):
        selection = self.df.iloc[idx]
        mean_v = np.array([s2_mean[i] for i in self.channels])
        std_v = np.array([s2_std[i] for i in self.channels])

        image_now = self.open_image(selection['s2_path'])
        image_now = image_now[self.channels, :, :]
        image_pre = self.open_image(selection['s2_three_month_path'])
        image_pre = image_pre[self.channels, :, :]
        image_prepre = self.open_image(selection['s2_year_path'])
        image_prepre = image_prepre[self.channels, :, :]
        normalized_image_now = (image_now - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]
        normalized_image_pre = (image_pre - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]
        normalized_image_prepre = (image_prepre - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]
        image = np.concatenate((normalized_image_prepre, normalized_image_pre, normalized_image_now), axis = 0)
        image = self.random_crop(image)
        x1 = self.transform(image)
        return x1
