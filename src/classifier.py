import torch
import torch.nn as nn
from torchvision import models

def create_resnet50_baseline(num_classes=2, pretrained=True, freeze_base=True):
    """
    Creates a ResNet50 model for baseline classification.

    Args:
        num_classes (int): Number of output classes (default: 2 for binary).
        pretrained (bool): Whether to load weights pre-trained on ImageNet.
        freeze_base (bool): Whether to freeze the weights of the base ResNet layers.

    Returns:
        torch.nn.Module: The configured ResNet50 model.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, num_classes)

    print(f"Created ResNet50 baseline. Pretrained: {pretrained}, Base Frozen: {freeze_base}, Output Classes: {num_classes}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model

if __name__ == '__main__':
    # Test case 1: Pretrained, frozen base
    print("--- Test Case 1: Pretrained, Frozen Base ---")
    model1 = create_resnet50_baseline(num_classes=2, pretrained=True, freeze_base=True)
    # print(model1)

    # Test case 2: Pretrained, unfrozen base (full fine-tuning)
    print("\n--- Test Case 2: Pretrained, Unfrozen Base ---")
    model2 = create_resnet50_baseline(num_classes=2, pretrained=True, freeze_base=False)
    # print(model2)

    # Test case 3: From scratch (random weights)
    print("\n--- Test Case 3: From Scratch ---")
    model3 = create_resnet50_baseline(num_classes=2, pretrained=False, freeze_base=False) # freeze_base is irrelevant if not pretrained
    # print(model3)

    try:
        print("\n--- Testing Forward Pass ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model1.to(device)
        dummy_input = torch.randn(4, 3, 224, 224).to(device)
        output = model1(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print("Forward pass successful.")
    except Exception as e:
        print(f"Error during forward pass test: {e}")
