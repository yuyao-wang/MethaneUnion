# Copyright 2020 DeepMind Technologies Limited.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for loading methane chip TIFF datasets."""

import enum
from pathlib import Path
from typing import Dict, Generator, Iterable, Mapping, Optional, Sequence, Text, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tifffile

Batch = Mapping[Text, np.ndarray]

_DEFAULT_DATA_DIR = Path('data/pretrain/data_download_chips_32')
_DEFAULT_SPLIT_FRACTIONS = {'train': 0.9, 'valid': 0.05, 'test': 0.05}

_CHANNEL_MEAN: Optional[np.ndarray] = None
_CHANNEL_STD: Optional[np.ndarray] = None


class Split(enum.Enum):
  """Dataset split."""
  TRAIN = 1
  TRAIN_AND_VALID = 2
  VALID = 3
  TEST = 4

  @classmethod
  def from_string(cls, name: Text) -> 'Split':
    return {
        'TRAIN': Split.TRAIN,
        'TRAIN_AND_VALID': Split.TRAIN_AND_VALID,
        'VALID': Split.VALID,
        'VALIDATION': Split.VALID,
        'TEST': Split.TEST,
    }[name.upper()]


class PreprocessMode(enum.Enum):
  """Preprocessing modes for the dataset."""
  PRETRAIN = 1
  LINEAR_TRAIN = 2
  EVAL = 3


def default_dataset_config() -> Dict[Text, object]:
  return dict(
      data_dir=str(_DEFAULT_DATA_DIR),
      image_size=128,
      num_channels=12,
      stats_path=str(_DEFAULT_DATA_DIR / 'channel_stats.npz'),
      split_fractions=dict(_DEFAULT_SPLIT_FRACTIONS),
      split_seed=0,
  )


def normalize_images(images: jnp.ndarray) -> jnp.ndarray:
  """Normalize the image using per-band statistics."""
  if _CHANNEL_MEAN is None or _CHANNEL_STD is None:
    raise ValueError('Normalization statistics are not initialized.')
  mean = jnp.asarray(_CHANNEL_MEAN).reshape((1, 1, 1, -1))
  std = jnp.asarray(_CHANNEL_STD).reshape((1, 1, 1, -1))
  return (images - mean) / (std + 1e-6)


def set_normalization_stats(mean: np.ndarray, std: np.ndarray):
  global _CHANNEL_MEAN, _CHANNEL_STD
  _CHANNEL_MEAN = mean.astype(np.float32)
  _CHANNEL_STD = std.astype(np.float32)


def count_examples(
    data_dir: Text,
    split: Split = Split.TRAIN_AND_VALID,
    split_fractions: Optional[Mapping[Text, float]] = None,
) -> int:
  files = _list_tif_files(Path(data_dir))
  total = len(files)
  if split == Split.TRAIN_AND_VALID:
    return total
  fractions = _merge_split_fractions(split_fractions)
  train_end, valid_end, test_end = _split_boundaries(total, fractions)
  if split == Split.TRAIN:
    return train_end
  if split == Split.VALID:
    return max(0, valid_end - train_end)
  assert split == Split.TEST
  return max(0, test_end - valid_end)


def load(
    split: Split,
    *,
    preprocess_mode: PreprocessMode,
    batch_dims: Sequence[int],
    dataset_config: Optional[Mapping[Text, object]] = None,
    transpose: bool = False,
    allow_caching: bool = False) -> Generator[Batch, None, None]:
  """Loads the given split of the methane chip dataset."""
  del transpose  # The double-transpose trick is disabled for TIFF inputs.
  del allow_caching

  config = _apply_dataset_defaults(dataset_config)
  all_files = _list_tif_files(Path(config['data_dir']))
  if not all_files:
    raise ValueError(
        f'No TIFF files were found under {config["data_dir"]}. '
        'Ensure the methane chips are downloaded.')

  stats_mean, stats_std = _load_or_compute_stats(
      all_files, config['num_channels'], config['stats_path'])
  set_normalization_stats(stats_mean, stats_std)

  split_files = _split_files(
      all_files, split, config['split_fractions'], config['split_seed'])
  split_files = _shard_files(split_files, jax.host_id(), jax.host_count())
  if not split_files:
    raise ValueError(
        'Shard for host '
        f'{jax.host_id()} is empty. Reduce the number of hosts or ensure '
        'there are more chips on disk.')

  total_batch_size = int(np.prod(batch_dims))
  iterator = (_infinite_batch_iterator
              if preprocess_mode is not PreprocessMode.EVAL
              else _finite_batch_iterator)
  for batch in iterator(
      split_files, total_batch_size, preprocess_mode, config):
    yield _reshape_batch(batch, batch_dims)


def transpose_images(batch: Batch):
  """Transpose images for TPU training."""
  new_batch = dict(batch)
  if 'images' in new_batch:
    new_batch['images'] = jnp.transpose(new_batch['images'], (3, 0, 1, 2))
  else:
    new_batch['view1'] = jnp.transpose(new_batch['view1'], (3, 0, 1, 2))
    new_batch['view2'] = jnp.transpose(new_batch['view2'], (3, 0, 1, 2))
  return new_batch


def _apply_dataset_defaults(
    dataset_config: Optional[Mapping[Text, object]],
) -> Dict[Text, object]:
  config = default_dataset_config()
  if dataset_config:
    for key, value in dataset_config.items():
      if key == 'split_fractions' and value is not None:
        merged = dict(_DEFAULT_SPLIT_FRACTIONS)
        merged.update(value)
        config[key] = merged
      else:
        config[key] = value
  config['data_dir'] = str(config['data_dir'])
  return config


def _list_tif_files(data_dir: Path) -> Sequence[Path]:
  return sorted(data_dir.rglob('*.tif'))


def _merge_split_fractions(
    fractions: Optional[Mapping[Text, float]]
) -> Dict[Text, float]:
  merged = dict(_DEFAULT_SPLIT_FRACTIONS)
  if fractions:
    merged.update(fractions)
  return merged


def _split_boundaries(
    total: int,
    fractions: Mapping[Text, float],
) -> Tuple[int, int, int]:
  train = int(total * fractions.get('train', 0.0))
  valid = int(total * fractions.get('valid', 0.0))
  test = int(total * fractions.get('test', 0.0))
  used = train + valid + test
  if used > total:
    overflow = used - total
    test = max(0, test - overflow)
    used = train + valid + test
  remainder = total - used
  train += remainder
  train = min(train, total)
  valid = min(train + valid, total)
  test = min(valid + test, total)
  return train, valid, test


def _split_files(
    files: Sequence[Path],
    split: Split,
    fractions: Mapping[Text, float],
    split_seed: int,
) -> Sequence[Path]:
  if split == Split.TRAIN_AND_VALID:
    return list(files)
  rng = np.random.default_rng(split_seed)
  perm = rng.permutation(len(files))
  shuffled = [files[i] for i in perm]
  train_end, valid_end, test_end = _split_boundaries(len(files), fractions)
  if split == Split.TRAIN:
    subset = shuffled[:train_end]
  elif split == Split.VALID:
    subset = shuffled[train_end:valid_end]
  else:
    subset = shuffled[valid_end:test_end]
  if not subset:
    return list(shuffled)
  return subset


def _shard_files(
    files: Sequence[Path],
    shard_index: int,
    num_shards: int,
) -> Sequence[Path]:
  if num_shards <= 1:
    return list(files)
  total = len(files)
  base = total // num_shards
  remainder = total % num_shards
  start = shard_index * base + min(shard_index, remainder)
  end = start + base + (1 if shard_index < remainder else 0)
  return files[start:end]


def _load_or_compute_stats(
    files: Sequence[Path],
    num_channels: int,
    stats_path: Optional[Text],
) -> Tuple[np.ndarray, np.ndarray]:
  if stats_path:
    stats_file = Path(stats_path)
    if stats_file.exists():
      data = np.load(stats_file)
      return data['mean'], data['std']
  mean, std = _compute_stats(files, num_channels)
  if stats_path:
    stats_file = Path(stats_path)
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(stats_file, mean=mean, std=std)
  return mean, std


def _compute_stats(
    files: Sequence[Path],
    num_channels: int,
) -> Tuple[np.ndarray, np.ndarray]:
  sum_channels = np.zeros(num_channels, dtype=np.float64)
  sum_sq_channels = np.zeros(num_channels, dtype=np.float64)
  total_pixels = 0
  for path in files:
    image = _read_image(path, num_channels)
    pixels = image.reshape(-1, image.shape[-1])
    sum_channels += pixels.sum(axis=0)
    sum_sq_channels += (pixels ** 2).sum(axis=0)
    total_pixels += pixels.shape[0]
  total_pixels = max(total_pixels, 1)
  mean = sum_channels / total_pixels
  variance = sum_sq_channels / total_pixels - mean ** 2
  std = np.sqrt(np.maximum(variance, 1e-6))
  return mean.astype(np.float32), std.astype(np.float32)


def _infinite_batch_iterator(
    files: Sequence[Path],
    batch_size: int,
    preprocess_mode: PreprocessMode,
    config: Mapping[Text, object],
) -> Iterable[Batch]:
  rng = np.random.default_rng(17 + jax.host_id())
  file_array = np.asarray(files)
  while True:
    indices = rng.integers(0, len(file_array), size=batch_size)
    batch_paths = file_array[indices]
    yield _batch_from_paths(batch_paths, preprocess_mode, config)


def _finite_batch_iterator(
    files: Sequence[Path],
    batch_size: int,
    preprocess_mode: PreprocessMode,
    config: Mapping[Text, object],
) -> Iterable[Batch]:
  total = len(files) // batch_size
  for batch_idx in range(total):
    start = batch_idx * batch_size
    end = start + batch_size
    batch_paths = files[start:end]
    yield _batch_from_paths(batch_paths, preprocess_mode, config)


def _batch_from_paths(
    paths: Sequence[Path],
    preprocess_mode: PreprocessMode,
    config: Mapping[Text, object],
) -> Batch:
  images = [_prepare_image(path, config) for path in paths]
  image_array = np.stack(images, axis=0)
  labels = np.zeros((len(paths),), dtype=np.int32)
  if preprocess_mode is PreprocessMode.PRETRAIN:
    return {
        'view1': image_array,
        'view2': np.copy(image_array),
        'labels': labels,
    }
  else:
    return {'images': image_array, 'labels': labels}


def _prepare_image(path: Path, config: Mapping[Text, object]) -> np.ndarray:
  image = _read_image(path, int(config['num_channels']))
  return _resize_image(image, int(config['image_size']))


def _read_image(path: Path, num_channels: int) -> np.ndarray:
  array = tifffile.imread(str(path)).astype(np.float32)
  if array.ndim == 2:
    array = array[..., np.newaxis]
  if array.ndim != 3:
    raise ValueError(f'Unsupported image dimensions for {path}: {array.shape}')
  if array.shape[-1] == num_channels:
    return array
  if array.shape[0] == num_channels:
    return np.moveaxis(array, 0, -1)
  raise ValueError(
      f'Image {path} has {array.shape[-1]} channels, expected {num_channels}')


def _resize_image(image: np.ndarray, target_size: int) -> np.ndarray:
  height, width, _ = image.shape
  if height == target_size and width == target_size:
    return image
  if (target_size % height) or (target_size % width):
    raise ValueError(
        f'Cannot upsample from {(height, width)} to {(target_size, target_size)} '
        'using integer strides.')
  scale_y = target_size // height
  scale_x = target_size // width
  image = np.repeat(image, scale_y, axis=0)
  return np.repeat(image, scale_x, axis=1)


def _reshape_batch(batch: Batch, batch_dims: Sequence[int]) -> Batch:
  reshaped = {}
  batch_dims = tuple(batch_dims)
  for key, value in batch.items():
    rest = value.shape[1:]
    reshaped[key] = value.reshape(batch_dims + rest)
  return reshaped
