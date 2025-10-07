import torch
import torch.nn as nn
import torchvision.models as models

def create_model(num_classes: int):
    """
    Create a neural network model for classification.
    Uses a pre-trained ResNet18 and replaces the final layer for num_classes output.
    """
    # Load a pre-trained ResNet18 model
    model = models.resnet18(weights=True)
    # Replace the final fully connected layer
    # The original ResNet18 has fc of size (512 -> 1000); we need (512 -> num_classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
