import math
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataset.MethaneGEEL2APretrainDataset import (
    MethaneGEEL2APretrainDataset,
    RSBYOLTransform,
    s2_mean,
    s2_std,
)
from utils import load_config, parse_args, set_all_seeds
torch.autograd.set_detect_anomaly(True)
mp.set_sharing_strategy("file_system")

# 修改后的ResNet模型，支持12通道输入
class ResNetModified(nn.Module):
    def __init__(self, base_model, num_channels=12):
        super(ResNetModified, self).__init__()
        self.backbone = base_model(pretrained=False)
        # CIFAR-style stem keeps 32x32 chips informative by avoiding early downsampling.
        self.backbone.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()  # 移除原有的全连接层

    def forward(self, x):
        x = self.backbone(x)
        return x

# 定义投影头
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048, output_dim=256):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x):
        x = self.net(x)
        return x

# 定义预测头（BYOL独有）
class PredictionHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=256):
        super(PredictionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.net(x)
        return x

# BYOL模型定义
class BYOL(nn.Module):
    def __init__(self, base_encoder, encoder_out_dim = 2048, num_channels=12, hidden_dim=2048, proj_dim=256, pred_dim=256, momentum=0.996):
        super(BYOL, self).__init__()
        # 在线网络
        self.online_encoder = ResNetModified(base_encoder, num_channels)
        self.online_projector = ProjectionHead(input_dim=encoder_out_dim, hidden_dim=hidden_dim, output_dim=proj_dim)
        self.online_predictor = PredictionHead(input_dim=proj_dim, hidden_dim=hidden_dim, output_dim=pred_dim)

        # 目标网络
        self.target_encoder = ResNetModified(base_encoder, num_channels)
        self.target_projector = ProjectionHead(input_dim=encoder_out_dim, hidden_dim=hidden_dim, output_dim=proj_dim)

        # 初始化目标网络参数
        self._initialize_target_network()

        self.base_momentum = momentum
        self.momentum = momentum  # 动量系数

    def _set_target_eval(self):
        self.target_encoder.eval()
        self.target_projector.eval()

    @torch.no_grad()
    def _initialize_target_network(self):
        # 将目标网络参数初始化为在线网络参数
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False  # 目标网络不需要梯度
        for param_o, param_t in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False
        self._set_target_eval()

    @torch.no_grad()
    def _update_target_network(self):
        # 更新目标网络参数
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1 - self.momentum)
        for param_o, param_t in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1 - self.momentum)
        self._set_target_eval()

    def forward(self, x1, x2):
        # 在线网络
        q1 = self.online_predictor(self.online_projector(self.online_encoder(x1).flatten(1)))
        q2 = self.online_predictor(self.online_projector(self.online_encoder(x2).flatten(1)))

        # 目标网络（不传梯度）
        with torch.no_grad():
            self._set_target_eval()
            k1 = self.target_projector(self.target_encoder(x1).flatten(1))
            k2 = self.target_projector(self.target_encoder(x2).flatten(1))

        # 计算损失
        loss1 = self.loss_fn(q1, k2)
        loss2 = self.loss_fn(q2, k1)
        loss = loss1 + loss2

        return loss

    def loss_fn(self, q, k):
        # BYOL损失函数，使用均方误差
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        return 2 - 2 * (q * k).sum(dim=-1).mean()

    def momentum_tau(self, cur_step: int, total_steps: int, base: float | None = None) -> float:
        if base is None:
            base = self.base_momentum
        total_steps = max(1, total_steps)
        return 1 - (1 - base) * (0.5 * (1 + math.cos(math.pi * cur_step / total_steps)))

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)

    if config['dp']:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        setup(rank, world_size)
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = config['device']

    os.makedirs(config['output_dir'], exist_ok=True)
    set_all_seeds(42)

    wandb.init(project='s2_pretrain')
    train_data_dir = config.get('data_dir')
    channel_indices = list(range(len(s2_mean)))
    train_transforms = RSBYOLTransform(
        size=32,
        mean=s2_mean[channel_indices],
        std=s2_std[channel_indices],
    )

    # 定义BYOL模型，使用ResNet18作为基模型
    if config['model'] == 'ResNet18':
        base_encoder = models.resnet18
        model = BYOL(base_encoder=base_encoder, encoder_out_dim = 512, num_channels=len(channel_indices)).to(device)
    elif config['model'] == 'ResNet34':
        base_encoder = models.resnet34
        model = BYOL(base_encoder=base_encoder, encoder_out_dim = 512, num_channels=len(channel_indices)).to(device)
    elif config['model'] == 'ResNet50':
        base_encoder = models.resnet50
        model = BYOL(base_encoder=base_encoder, encoder_out_dim = 2048, num_channels=len(channel_indices)).to(device)
    elif config['model'] == 'ResNet101':
        base_encoder = models.resnet101
        model = BYOL(base_encoder=base_encoder, encoder_out_dim = 2048, num_channels=len(channel_indices)).to(device)

    # Dataset and DataLoader
    train_dataset = MethaneGEEL2APretrainDataset(
        transform=train_transforms,
        data_dir=train_data_dir,
        channels=channel_indices,
    )
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16, drop_last=True)

    num_workers = config.get('num_workers', 2)
    pin_memory = config.get('pin_memory', True)
    prefetch_factor = config.get('prefetch_factor', 4)
    persistent_workers = config.get('persistent_workers', True) and num_workers > 0

    dataloader_kwargs = dict(
        batch_size=config['batch_size'],
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
        dataloader_kwargs["persistent_workers"] = persistent_workers

    if config['dp']:
        model = DDP(model, device_ids=[rank], broadcast_buffers=False, find_unused_parameters=True)
        train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=rank)
        train_loader = DataLoader(train_dataset, sampler=train_sampler, **dataloader_kwargs)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    byol_model = model.module if config['dp'] else model

    # 优化器和学习率调度器
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-6)

    num_epochs = config['num_epochs']

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6,
                                                           last_epoch=-1)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, num_epochs * steps_per_epoch)

    best_loss = 1000.0
    epoch_bar = tqdm(range(num_epochs), desc="Epochs", leave=True)
    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0

        if config['dp']:
            train_loader.sampler.set_epoch(epoch)

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False) as pbar:
            for idx, (x_i, x_j) in enumerate(train_loader):
                x1, x2 = x_i.float().to(device), x_j.float().to(device)
                # x1 = F.interpolate(x1, size=(32, 32), mode='bilinear', align_corners=False)
                # x2 = F.interpolate(x2, size=(32, 32), mode='bilinear', align_corners=False)

                optimizer.zero_grad()
                loss = model(x1, x2)

                loss.backward()
                optimizer.step()

                # 更新目标网络参数
                with torch.no_grad():
                    current_step = epoch * steps_per_epoch + idx
                    byol_model.momentum = byol_model.momentum_tau(current_step, total_steps)
                    byol_model._update_target_network()
                    embeddings = F.normalize(
                        byol_model.online_projector(byol_model.online_encoder(x1).flatten(1)),
                        dim=-1,
                    )
                    std_mean = embeddings.std(dim=0).mean().item()
                wandb.log({"train/loss": loss.item(), "z_std_mean": std_mean}, commit=False)

                total_loss += loss.item()
                pbar.set_postfix(loss=total_loss / (idx + 1))
                pbar.update(1)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch, "training loss": avg_loss, "lr": scheduler.get_last_lr()[0]})
        if not config['dp'] or dist.get_rank() == 0:
            backbone_state = byol_model.online_encoder.backbone.state_dict()
            if epoch % config['save_epoch'] == 0:
                model_save_path = os.path.join(config['output_dir'], "epoch_{}.pth".format(epoch))
                torch.save({
                    'backbone_state': backbone_state,
                    'epoch': epoch,
                    'avg_loss': avg_loss,
                }, model_save_path)
            if avg_loss < best_loss:
                best_loss = avg_loss
                model_save_path = os.path.join(config['output_dir'], "best.pth")
                torch.save({
                    'backbone_state': backbone_state,
                    'epoch': epoch,
                    'avg_loss': avg_loss,
                }, model_save_path)
        scheduler.step()
    print(f'Training is done')
    if config['dp']:
        cleanup()
