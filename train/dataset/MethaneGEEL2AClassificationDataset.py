import numpy as np
from torch.utils.data import Dataset
import logging
import torch
import pandas as pd
import rasterio
import tifffile
import random
import scipy.ndimage
import wandb
import time
import cv2
from pathlib import Path

logger = logging.getLogger(__name__)
s2_mean = [855.06971511, 
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
    ]
s2_std = [567.3335958, 
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
    ]

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
        channels, height, width = sample['image'].shape
        if random.random() < 1:
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
    
class Resize:
    def __init__(self, target_size=(32, 32)):
        self.target_size = target_size

    def __call__(self, sample):
        # 处理浮点数据的每个通道
        image = np.array([cv2.resize(channel, self.target_size, interpolation=cv2.INTER_LINEAR).astype(np.float32) for channel in sample['image']])
        # 处理标签的最近邻插值
        label = cv2.resize(sample['label'], self.target_size, interpolation=cv2.INTER_NEAREST).astype(np.float32)
        return {'image': image, 'label': label}

class ResizeTensor:
    def __init__(self, target_size=(32, 32)):
        self.target_size = target_size

    def __call__(self, sample):
        # Convert image and label to PyTorch tensors and send to GPU
        image = torch.tensor(sample['image'], dtype=torch.float32).unsqueeze(0).cuda()  # Add batch dimension
        label = torch.tensor(sample['label'], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()  # Add batch and channel
        
        # Resize using bilinear and nearest for image and label respectively
        resized_image = F.interpolate(image, size=self.target_size, mode='bilinear', align_corners=False)
        resized_label = F.interpolate(label, size=self.target_size, mode='nearest')
        
        return {'image': resized_image.squeeze(0).cpu().numpy(), 'label': resized_label.squeeze().cpu().numpy()}
    
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


class MethaneGEEL2AClassificationDataset(Dataset):
    def __init__(self, transform, data_path, channels, data_range = 'now', location_range = 'all', image_root = '/home/ruoyu/methane_emission/data/MethaneS2CM/l2a_temporal_split_32x32_bandsProcessed'):
        
        self.df = pd.read_csv(data_path)
        print(f'data path: {data_path} length of df {len(self.df)}')
        self.transform = transform
        self.channels = channels
        self.data_range = data_range
        self.epsilon = 1e-9
        # Root directory used for resolving relative chip paths
        self.image_root = Path(image_root) if image_root is not None else Path(data_path).parent
        if location_range == 'permian':
            lat_min, lat_max = 31, 33
            lon_min, lon_max = -104, -101

            # lat_min, lat_max = 30, 35
            # lon_min, lon_max = -105, -100

            # 过滤在范围内的行
            self.df = self.df[(self.df['latitude'] >= lat_min) & (self.df['latitude'] <= lat_max) & 
                            (self.df['longitude'] >= lon_min) & (self.df['longitude'] <= lon_max)]
        elif location_range == 'algeria':
            lat_min, lat_max = 18.96, 37.09
            lon_min, lon_max = -8.67, 11.98

            # 过滤在范围内的行
            self.df = self.df[(self.df['latitude'] >= lat_min) & (self.df['latitude'] <= lat_max) & 
                            (self.df['longitude'] >= lon_min) & (self.df['longitude'] <= lon_max)]
        elif location_range == 'turkmenistan':
            lat_min, lat_max = 35.13, 42.79
            lon_min, lon_max = 52.44, 66.68

            # 过滤在范围内的行
            self.df = self.df[(self.df['latitude'] >= lat_min) & (self.df['latitude'] <= lat_max) & 
                            (self.df['longitude'] >= lon_min) & (self.df['longitude'] <= lon_max)]
        print(f'positive samples ratio {len(self.df[self.df['label'] == 1])} / {len(self.df)}')
    def __len__(self):
        return len(self.df)
    
    def _resolve_image_path(self, img_path):
        path = Path(img_path)
        if not path.is_absolute():
            candidate = self.image_root / path
            return candidate
        return path

    def open_image(self, img_path):
        resolved_path = self._resolve_image_path(img_path)
        img = tifffile.imread(resolved_path)
        return img.astype(np.float32)

    def __getitem__(self, idx):
        start_time = time.time()
        selection = self.df.iloc[idx]
        mean_v = np.array([s2_mean[i] for i in self.channels])
        std_v = np.array([s2_std[i] for i in self.channels])

        if self.data_range == 'now':
            image_now = self.open_image(selection['s2_path'])
            image_now = image_now[self.channels, :, :]
            normalized_image_now = (image_now - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]
            image = normalized_image_now
        elif self.data_range == 'now+pre':
            image_now = self.open_image(selection['s2_path'])
            image_now = image_now[self.channels, :, :]
            image_pre = self.open_image(selection['s2_pre_path'])
            image_pre = image_pre[self.channels, :, :]
            normalized_image_now = (image_now - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]
            normalized_image_pre = (image_pre - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]
            image = np.concatenate((normalized_image_pre, normalized_image_now), axis = 0)
        elif self.data_range == 'now+pre+post':
            image_now = self.open_image(selection['s2_path'])
            image_now = image_now[self.channels, :, :]
            image_pre = self.open_image(selection['s2_pre_path'])
            image_pre = image_pre[self.channels, :, :]
            image_post = self.open_image(selection['s2_post_path'])
            image_post = image_post[self.channels, :, :]
            normalized_image_now = (image_now - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]
            normalized_image_pre = (image_pre - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]
            normalized_image_post = (image_post - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]
            image = np.concatenate((normalized_image_pre, normalized_image_now, normalized_image_post), axis = 0)
        elif self.data_range == 'now+pre+prepre':
            image_now = self.open_image(selection['s2_path'])
            image_now = image_now[self.channels, :, :]
            image_pre = self.open_image(selection['s2_pre_path'])
            image_pre = image_pre[self.channels, :, :]
            image_prepre = self.open_image(selection['s2_pre_pre_path'])
            image_prepre = image_prepre[self.channels, :, :]
            # normalized_image_now = (image_now - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]
            # normalized_image_pre = (image_pre - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]
            # normalized_image_prepre = (image_prepre - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]
            # image = np.concatenate((normalized_image_prepre, normalized_image_pre, normalized_image_now), axis = 0)

            normalized_image_now = image_now
            normalized_image_pre = image_pre
            normalized_image_prepre = image_prepre
            image = np.concatenate((normalized_image_prepre, normalized_image_pre, normalized_image_now), axis = 0)
        elif self.data_range == 'now+pre+prepre+before+after':
            image_now = self.open_image(selection['s2_path'])
            image_now = image_now[self.channels, :, :]
            image_pre = self.open_image(selection['s2_pre_path'])
            image_pre = image_pre[self.channels, :, :]
            image_prepre = self.open_image(selection['s2_pre_pre_path'])
            image_prepre = image_prepre[self.channels, :, :]
            # "s2_before_path": nt3_path, "s2_after_path"
            image_before = self.open_image(selection['s2_before_path'])
            image_before = image_before[self.channels, :, :]
            image_after = self.open_image(selection['s2_after_path'])
            image_after = image_after[self.channels, :, :]

            normalized_image_now = (image_now - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]
            normalized_image_pre = (image_pre - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]
            normalized_image_prepre = (image_prepre - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]
            normalized_image_before = (image_before - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]
            normalized_image_after = (image_after - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]

            image = np.concatenate((normalized_image_prepre, normalized_image_pre, normalized_image_before, normalized_image_now, normalized_image_after), axis = 0)
        elif self.data_range == 'now+pre+post+fft':
            image_now = self.open_image(selection['s2_path'])
            image_now = image_now[self.channels, :, :]
            image_pre = self.open_image(selection['s2_pre_path'])
            image_pre = image_pre[self.channels, :, :]
            image_post = self.open_image(selection['s2_post_path'])
            image_post = image_post[self.channels, :, :]
            normalized_image_now = (image_now - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]
            normalized_image_pre = (image_pre - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]
            normalized_image_post = (image_post - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]
            image = np.concatenate((normalized_image_pre, normalized_image_now, normalized_image_post), axis = 0)
            magnitude_spectrum = np.zeros_like(image)
            for i in range(image.shape[0]):
                f_transform = np.fft.fft2(image[i])
                f_transform_shifted = np.fft.fftshift(f_transform)
                magnitude_spectrum[i] = 20 * np.log(np.abs(f_transform_shifted) + 1)  # To avoid log(0)

            # Normalize both images
            image = image / 255.0
            magnitude_spectrum = magnitude_spectrum / np.max(magnitude_spectrum, axis=(1,2), keepdims=True)

            # Stack the original image and the magnitude spectrum
            image = np.concatenate((image, magnitude_spectrum), axis=0)  # Shape: (54, 96, 96)
        elif self.data_range == 'ndmi':
            image_now = self.open_image(selection['s2_path'])
            image_now = image_now[self.channels, :, :]
            image_pre = self.open_image(selection['s2_pre_path'])
            image_pre = image_pre[self.channels, :, :]
            normalized_image_now = (image_now - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]
            normalized_image_pre = (image_pre - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]
            image_list = []
            for i in range(len(self.channels) - 2):
                image_list.append(image_now[i] / (image_now[-2] + self.epsilon) - image_pre[i] / (image_pre[-2] + self.epsilon))
            for i in range(len(self.channels) - 1):
                image_list.append(image_now[i] / (image_now[-1] + self.epsilon) - image_pre[i] / (image_pre[-1] + self.epsilon))
            image = np.stack(image_list)
        elif self.data_range == 'b13':
            image_now = self.open_image(selection['s2_path'])
            image_now = image_now[self.channels, :, :]
            image_pre = self.open_image(selection['s2_pre_path'])
            image_pre = image_pre[self.channels, :, :]
            normalized_image_now = (image_now - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]
            normalized_image_pre = (image_pre - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]
            # image = np.concatenate((normalized_image_pre, normalized_image_now, normalized_image_now[-1:,:,:]), axis = 0)
            image = normalized_image_now
        elif self.data_range == 'all_time':
            image_keys = ['s2_path', 's2_pre_path', 's2_pre_pre_path']
            valid_keys = [key for key in image_keys if isinstance(selection[key], str) and selection[key]]
            if not valid_keys:
                # fall back to current observation if metadata is missing
                valid_keys = ['s2_path']
            chosen_key = random.choice(valid_keys)
            image_all_time = self.open_image(selection[chosen_key])
            image_all_time = image_all_time[self.channels, :, :]
            image = (image_all_time - mean_v[:, np.newaxis, np.newaxis]) / std_v[:, np.newaxis, np.newaxis]

        label = self.open_image(selection['plume_mask_path'])
        load_time = time.time()
        sample = {'image': image, 'label': label}
        sample = self.transform(sample)
        sample['mask'] = sample['label']
        sample['label'] = torch.tensor(selection['label'])
        sample['emission_auto'] = torch.tensor(selection['emission_auto'])
        transform_time = time.time()
        # print(f"Load time: {load_time - start_time}, Transform time: {transform_time - load_time}")
        return sample
