import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

def get_data_loaders(data_dir, batch_size=64):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, len(train_dataset.classes)

def get_class_names(data_dir):

    train_dir = os.path.join(data_dir, 'train')
    if os.path.exists(train_dir):
        class_names = sorted([d for d in os.listdir(train_dir) 
                            if os.path.isdir(os.path.join(train_dir, d))])
        return class_names
    return None