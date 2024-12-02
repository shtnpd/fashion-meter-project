import torch
from torchvision import models
from torch import nn

def create_custom_model(num_classes, device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)