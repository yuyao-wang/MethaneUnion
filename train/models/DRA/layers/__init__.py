import torch
from models.DRA.layers.deviation_loss import DeviationLoss
from models.DRA.layers.binary_focal_loss import BinaryFocalLoss

def build_criterion(criterion, device):
    if criterion == "deviation":
        print("Loss : Deviation")
        return DeviationLoss(device)
    elif criterion == "BCE":
        print("Loss : Binary Cross Entropy")
        return torch.nn.BCEWithLogitsLoss()
    elif criterion == "focal":
        print("Loss : Focal")
        return BinaryFocalLoss()
    elif criterion == "CE":
        print("Loss : CE")
        return torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError