import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from model import myCNN
from data import get_class_names, get_data_loaders


def load_model(model_path, num_classes, device):

    model = myCNN(num_classes=num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("model weights loaded successfully")
    return model

def train_epoch(model, loader, criterion, optimizer, device):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

def train_model(data_dir, num_epochs=50, batch_size=32, lr=0.001, resume_from_checkpoint=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")
    


    train_loader, val_loader, num_classes = get_data_loaders(data_dir, batch_size)
    print(f"number of classes: {num_classes}")
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")
    class_names = get_class_names(data_dir)
    print(f"classes: {class_names}\n")
    
    model_path = 'final_model.pth'
    
    if resume_from_checkpoint and os.path.exists(model_path):
        print(f"resuming from checkpoint: {model_path}")
        try:
            model = load_model(model_path, num_classes, device)
            print("training will continue from saved weights\n")
        except Exception as e:
            print(f"failed to load checkpoint: {e}")
            print("starting fresh training instead\n")
            model = myCNN(num_classes=num_classes).to(device)
    else:
        print("starting fresh training\n")
        model = myCNN(num_classes=num_classes).to(device)
    

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    best_acc = 0.0
    print("Starting training...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        print(f"epoch {epoch+1}/{num_epochs}")
        print(f" train Loss: {train_loss:.4f}, train Acc: {train_acc:.2f}%")
        print(f"  val Loss: {val_loss:.4f}, val Acc: {val_acc:.2f}%")
        

        if val_acc > best_acc:
            best_acc = val_acc

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'num_classes': num_classes,
                'class_names': class_names
            }, model_path)
            
            print(f"saved best model with accuracy: {best_acc:.2f}%")
        
        print("-" * 60)
    
    print(f"\ntraining complete! Best validation accuracy: {best_acc:.2f}%")
    return model, best_acc

if __name__ == "__main__":

    data_dir = "brands"
    if not os.path.exists(data_dir):
        print(f"error: Data directory '{data_dir}' not found!")
        exit(1)
    
    if not os.path.exists(os.path.join(data_dir, 'train')):
        print(f"error: '{data_dir}/train' directory not found!")
        exit(1)
    
    if not os.path.exists(os.path.join(data_dir, 'val')):
        print(f"error: '{data_dir}/val' directory not found!")
        exit(1)
    
    train_model(
        data_dir=data_dir,
        num_epochs=100,
        batch_size=64,
        lr=0.0001,
        resume_from_checkpoint=False 
    )