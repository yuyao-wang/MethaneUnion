import os
import numpy as np
import torch
from torch import nn, optim
# from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, random_split, RandomSampler
from torch.utils.tensorboard import SummaryWriter
import argparse
from torchmetrics import JaccardIndex, Accuracy, Precision, Recall, F1Score

import time
import random

# from models.unet import UNet
from torchvision import transforms
from tqdm import tqdm

import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils import parse_args, load_config
# from models.transformer_unet import TransformerUNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset.MethaneGEEL2AClassificationDataset import *
# from models.attention_cnn import AttentionCNN
# from models.csa_classifier import CSAClassifier
from models.resnet import ResNet18, ResNet50, ResNet101, ResNet34
from sklearn.metrics import roc_auc_score
from models.vit_classifier import VisionTransformer
from models.FreResNet import frequency_resnet18
from models.FcaNet import fcanet18
from timm.models import create_model
import models.spectformer
from models.SEResNet import seresnet18

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, reduction='mean'):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if reduction == 'mean':
            return torch.mean(F_loss)
        elif reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        
class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, default_reduction='mean'):
        super(CustomCrossEntropyLoss, self).__init__()
        self.default_reduction = default_reduction
        self.class_weights = torch.tensor([0.3, 0.7]).to(device)

    def forward(self, input, target, reduction=None):
        if reduction is None:
            reduction = self.default_reduction
        return F.cross_entropy(input, target, reduction=reduction)
    
def set_all_seeds(seed):
    """Set all the seeds to fix the same initial conditions for each training;
    :param seed: seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def compute_metrics(preds, targets):
    # print(f'preds: {preds}')
    # print(f'targets: {targets}')
    tp = (preds * targets).sum().item()
    tn = ((1 - preds) * (1 - targets)).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    correct = 0
    # criterion = FocalLoss().to(device)
    # criterion = CustomCrossEntropyLoss().to(device)
    criterion = nn.BCEWithLogitsLoss()
    with tqdm(enumerate(train_loader), unit="batch", total=len(train_loader)) as tepoch:
        for batch_idx, batch in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            data = batch['image'].float().to(device)
            target = batch['label'].float().to(device)
            optimizer.zero_grad()
            output = model(data).squeeze()
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # Translated comment
            loss.backward()
            optimizer.step()
            
            # Translated comment
            probs = torch.sigmoid(output)
            pred = (probs >= 0.5).float()
            
            # Translated comment
            correct += pred.eq(target).sum().item()
            
            # Translated comment
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            tepoch.set_postfix(loss_epoch=loss.item())
        # tepoch.set_postfix(loss_epoch=total_loss / len(train_loader))
    accuracy, precision, recall, f1_score = compute_metrics(torch.tensor(all_preds), torch.tensor(all_targets))
    print(f'\nTrain Epoch: {epoch}: Average loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}, '
              f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}\n')
    # print(f'Train Epoch: {epoch}  loss: {total_loss / len(train_loader)}')
    wandb.log({"epoch": epoch, "training loss": total_loss / len(train_loader), "Train Accuracy": accuracy, "Train Precision": precision, "Train Recall": recall, "Train F1 Score": f1_score}, step = epoch)

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    # criterion = FocalLoss().to(device)
    # criterion = CustomCrossEntropyLoss().to(device)
    criterion = nn.BCEWithLogitsLoss()
    accuracy, precision, recall, f1_score = 0.0, 0.0, 0.0, 0.0
    if not config['dp'] or dist.get_rank() == 0:
        all_preds = []
        all_targets = []
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                data = batch['image'].float().to(device)
                target = batch['label'].float().to(device)
                output = model(data).squeeze()
                test_loss += criterion(output, target).item()
        
                # Translated comment
                probs = torch.sigmoid(output)
                pred = (probs >= 0.5).float()  # Translated comment
                
                correct += pred.eq(target).sum().item()
                
                # Translated comment
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.append(probs)
                all_labels.append(target)

        test_loss /= len(test_loader.dataset)
        accuracy, precision, recall, f1_score = compute_metrics(torch.tensor(all_preds), torch.tensor(all_targets))
        auc = roc_auc_score(torch.cat(all_labels).cpu().numpy(), torch.cat(all_probs).cpu().numpy())
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, '
              f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}, AUC: {auc}\n')
        wandb.log({"test loss": test_loss, "Test Accuracy": accuracy, "Test Precision": precision, "Test Recall": recall, "Test F1 Score": f1_score, "Test AUC": auc}, step = epoch)
    return accuracy, precision, recall, f1_score

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def enable_linear_probe(model):
    """Freeze backbone weights and only allow the classification head to update."""
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'fc'):
        for param in model.backbone.fc.parameters():
            param.requires_grad = True
    else:
        raise ValueError('Linear probe currently supports ResNet-style models with backbone.fc')

################## MAIN ##################
if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    os.makedirs(config['output_dir'], exist_ok=True)
    if config['dp']:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        setup(rank, world_size)
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = config['device']
    
    wandb.init(project='s2_classification')
    set_all_seeds(42)
    
    if config['model'] == 'ResNet18':
        model = ResNet18(input_channel = config['input_channel'], num_classes = 1)
    elif config['model'] == 'ResNet50':
        model = ResNet50(input_channel = config['input_channel'], num_classes = 1)
    elif config['model'] == 'ResNet34':
        model = ResNet34(input_channel = config['input_channel'], num_classes = 1)
    elif config['model'] == 'ResNet101':
        model = ResNet101(input_channel = config['input_channel'], num_classes = 1)

    state_dict = torch.load(config['model_path'])
    backbone_state_dict = {k[len('online_encoder.'):]: v for k, v in state_dict.items() if k.startswith('online_encoder.backbone.')}
    model.load_state_dict(backbone_state_dict, strict = False)

    enable_linear_probe(model)

    model.to(device)
    
    # trainning parameters
    epochs = config['epochs']
    lr = config['lr']
    bs = config['batch_size']
    
    # initialize loss function
    # loss = DiceLoss()
    # loss = nn.BCEWithLogitsLoss()
    # loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # initialize optimizer
    loss = nn.CrossEntropyLoss()
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.AdamW(trainable_parameters, lr, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['lr_gamma'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_data_path = config['train_data_path']
    test_data_path = config['test_data_path']

    train_data_transforms = transforms.Compose([
            Resize((32, 32)),
            # RandomCrop((96, 96)),
            RandomRotation(),
            # RandomFlip(),
            # CenterCrop((96, 96)),
            ToTensor()
           ])

    test_data_transforms = transforms.Compose([
            Resize((32, 32)),
            # RandomCrop((96, 96)),
            ToTensor()
           ])
    
    data_train = MethaneGEEL2AClassificationDataset(train_data_transforms, train_data_path, config['channels'], config['data_range'], location_range=config['location_range'])
    data_val = MethaneGEEL2AClassificationDataset(test_data_transforms, test_data_path, config['channels'], config['data_range'], location_range=config['location_range'])

    if config['dp']:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        train_sampler = DistributedSampler(data_train, num_replicas=dist.get_world_size(), rank=rank)
    # initialize data loaders
        train_dl = DataLoader(data_train, batch_size=config['batch_size'], num_workers=8, sampler = train_sampler)
    else:
        train_dl = DataLoader(data_train, batch_size=config['batch_size'], num_workers=8, shuffle = True, pin_memory=True, prefetch_factor=4)
    val_dl = DataLoader(data_val, batch_size=config['batch_size'], num_workers=8, pin_memory=True, prefetch_factor=4)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train(model, device, train_dl, optimizer, epoch)
        accuracy, precision, recall, f1_score = test(model, device, val_dl, epoch)
        if epoch > 5:
            scheduler.step()
        print(f'current learning rate {scheduler.get_last_lr()[0]}')
        wandb.log({"lr": scheduler.get_last_lr()[0]}, step=epoch)
        if accuracy > best_acc:
            best_acc = accuracy
            file_path = os.path.join(config['output_dir'], 'best_model.pth')
            torch.save(model.state_dict(), file_path)
    if config['dp']:
        cleanup()
