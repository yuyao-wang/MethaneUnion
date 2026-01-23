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
"""Tests for BYOL's main training loop."""

import os
from pathlib import Path

from absl import flags
from absl.testing import absltest
import numpy as np
import tifffile

from byol import byol_experiment
from byol import eval_experiment
from byol import main_loop
from byol.configs import byol as byol_config
from byol.configs import eval as eval_config


FLAGS = flags.FLAGS


class MainLoopTest(absltest.TestCase):

  def _create_fake_dataset(self, num_files: int = 64) -> str:
    data_dir = self.create_tempdir(name='chips').full_path
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    image = np.random.rand(32, 32, 12).astype(np.float32)
    for idx in range(num_files):
      tifffile.imwrite(root / f'sample_{idx}.tif', image)
    return data_dir

  def _dataset_config(self, data_dir: str) -> dict:
    return dict(
        data_dir=data_dir,
        image_size=32,
        num_channels=12,
        stats_path=os.path.join(data_dir, 'stats.npz'),
        split_fractions={'train': 1.0, 'valid': 0.0, 'test': 0.0},
    )

  def test_pretrain(self):
    data_dir = self._create_fake_dataset()
    data_cfg = self._dataset_config(data_dir)
    config = byol_config.get_config(
        num_epochs=40, batch_size=4, dataset_config=data_cfg)
    temp_dir = self.create_tempdir().full_path

    # Override some config fields to make test lighter.
    config['network_config']['encoder_class'] = 'TinyResNet'
    config['network_config']['projector_hidden_size'] = 256
    config['network_config']['predictor_hidden_size'] = 256
    config['checkpointing_config']['checkpoint_dir'] = temp_dir
    config['evaluation_config']['batch_size'] = 8
    config['max_steps'] = 16

    experiment_class = byol_experiment.ByolExperiment
    main_loop.train_loop(experiment_class, config)
    main_loop.eval_loop(experiment_class, config)

  def test_linear_eval(self):
    data_dir = self._create_fake_dataset()
    data_cfg = self._dataset_config(data_dir)
    config = eval_config.get_config(
        checkpoint_to_evaluate=None, batch_size=4, dataset_config=data_cfg)
    temp_dir = self.create_tempdir().full_path

    # Override some config fields to make test lighter.
    config['network_config']['encoder_class'] = 'TinyResNet'
    config['allow_train_from_scratch'] = True
    config['checkpointing_config']['checkpoint_dir'] = temp_dir
    config['evaluation_config']['batch_size'] = 8
    config['max_steps'] = 16

    experiment_class = eval_experiment.EvalExperiment
    main_loop.train_loop(experiment_class, config)
    main_loop.eval_loop(experiment_class, config)

  def test_pipeline(self):
    data_dir = self._create_fake_dataset()
    data_cfg = self._dataset_config(data_dir)
    b_config = byol_config.get_config(
        num_epochs=40, batch_size=4, dataset_config=data_cfg)
    temp_dir = self.create_tempdir().full_path

    # Override some config fields to make test lighter.
    b_config['network_config']['encoder_class'] = 'TinyResNet'
    b_config['network_config']['projector_hidden_size'] = 256
    b_config['network_config']['predictor_hidden_size'] = 256
    b_config['checkpointing_config']['checkpoint_dir'] = temp_dir
    b_config['evaluation_config']['batch_size'] = 8
    b_config['max_steps'] = 16

    main_loop.train_loop(byol_experiment.ByolExperiment, b_config)

    e_config = eval_config.get_config(
        checkpoint_to_evaluate=f'{temp_dir}/pretrain.pkl',
        batch_size=4,
        dataset_config=data_cfg)

    # Override some config fields to make test lighter.
    e_config['network_config']['encoder_class'] = 'TinyResNet'
    e_config['allow_train_from_scratch'] = True
    e_config['checkpointing_config']['checkpoint_dir'] = temp_dir
    e_config['evaluation_config']['batch_size'] = 8
    e_config['max_steps'] = 16

    main_loop.train_loop(eval_experiment.EvalExperiment, e_config)


if __name__ == '__main__':
  absltest.main()
