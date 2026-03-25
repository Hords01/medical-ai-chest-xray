"""
Representative Training Pipeline
=================================
This is an illustrative implementation showing the training methodology
used in the chest X-ray classification project. The actual deployed code
remains with KTÜ.

Architecture: Transfer learning with frozen backbone + retrained classifier
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# ============================================================
# 1. CONFIGURATION
# ============================================================

CONFIG = {
    # Data settings
    "image_size": 299,
    "batch_size": 32,
    "num_classes": 2,  # Binary: Pathology Present / Absent

    # Training settings (fine-tuning phase)
    "epochs": 7,
    "learning_rate": 0.0001,
    "optimizer": "adam",
    "criterion": "cross_entropy",

    # Reproducibility
    "seed": 42,

    # Paths (example — actual paths would point to ethics-approved data)
    "train_dir": "data/train/",
    "test_dir": "data/test/",
}


def set_seed(seed):
    """Ensure reproducibility across runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. DATA TRANSFORMS
# ============================================================

def get_transforms(fine_tuning=True):
    """
    Two transform sets were used:
    - Without fine-tuning: simple resize + normalize
    - With fine-tuning: augmentation pipeline

    Augmentation rationale: With only ~809 training images,
    augmentation was critical to prevent overfitting.
    """
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                      std=[0.5, 0.5, 0.5])

    if fine_tuning:
        train_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.CenterCrop((299, 224)),
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(299, scale=(0.8, 1.0), ratio=(0.5, 1.333)),
            transforms.RandomHorizontalFlip(),
            transforms.Pad((60, 20, 60, 20), fill=0, padding_mode='constant'),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            normalize,
        ])

    test_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, test_transform


# ============================================================
# 3. MODEL SETUP
# ============================================================

def create_model(arch_name, num_classes, pretrained=True, freeze_backbone=True):
    """
    Load a pretrained model and modify the classifier for binary classification.

    Key decision: Freezing all layers except the final classifier.
    With only ~809 training images, retraining the full network would
    cause severe overfitting. The pretrained features from ImageNet
    provide a strong foundation for medical image features.
    """
    # Example for DenseNet201
    if arch_name == "densenet201":
        model = models.densenet201(weights="IMAGENET1K_V1" if pretrained else None)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        # Replace classifier for binary task
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)

    elif arch_name == "efficientnet":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)

    elif arch_name == "resnet101":
        model = models.resnet101(weights="IMAGENET1K_V1" if pretrained else None)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

    return model


# ============================================================
# 4. TRAINING LOOP
# ============================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ============================================================
# 5. METRICS — NPV CALCULATION
# ============================================================

def calculate_npv(y_true, y_pred):
    """
    Negative Predictive Value: TN / (TN + FN)

    Why NPV matters in clinical screening:
    When the model says "no pathology," how much can we trust it?
    A high NPV means fewer sick patients misclassified as healthy.
    """
    cm = confusion_matrix(y_true, y_pred)
    # For binary: cm[0][0] = TN, cm[1][0] = FN
    tn = cm[0][0]
    fn = cm[1][0]
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return npv


# ============================================================
# 6. MAIN TRAINING FLOW
# ============================================================

def main():
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_transform, test_transform = get_transforms(fine_tuning=True)
    train_dataset = datasets.ImageFolder(CONFIG["train_dir"], transform=train_transform)
    test_dataset = datasets.ImageFolder(CONFIG["test_dir"], transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                              shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"],
                             shuffle=False)

    # Models to train
    architectures = ["densenet201", "efficientnet", "resnet101"]

    for arch in architectures:
        print(f"\n{'='*60}")
        print(f"Training: {arch}")
        print(f"{'='*60}")

        model = create_model(arch, CONFIG["num_classes"]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=CONFIG["learning_rate"]
        )

        # Training
        for epoch in range(CONFIG["epochs"]):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            print(f"  Epoch {epoch+1}/{CONFIG['epochs']} — "
                  f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        # Evaluation
        test_loss, test_acc, preds, labels, probs = evaluate(
            model, test_loader, criterion, device
        )
        npv = calculate_npv(labels, preds)

        print(f"\n  Test Accuracy: {test_acc:.4f}")
        print(f"  NPV: {npv:.4f}")
        print(f"\n{classification_report(labels, preds)}")

        # Save model (in practice — not included in this repo)
        # torch.save(model.state_dict(), f"models/{arch}_pathology.pth")


if __name__ == "__main__":
    main()
