from model import myCNN
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import os

def load_model(model_path, num_classes, device):

    model = myCNN(num_classes=num_classes).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    

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

# we get the classes aka the cars brands
def get_class_names(data_dir):

    train_dir = os.path.join(data_dir, 'train')
    if os.path.exists(train_dir):
        class_names = sorted([d for d in os.listdir(train_dir) 
                            if os.path.isdir(os.path.join(train_dir, d))])
        return class_names
    return None

# we process the input image

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  
    return image

# a simple predict funtion to use and infrence our model
def predict(model, image_tensor, class_names, device):

    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100
    
    return predicted_class, confidence_score, probabilities[0]




model_path = 'best_car_model.pth' #THE CHOSEN MODEL
data_dir = 'brands' #DATA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

class_names = get_class_names(data_dir)

if class_names is None:
    print(f"error: Could not find class names in {data_dir}/train")
    exit(1)

num_classes = len(class_names)
print(f"nlasses found: {class_names}")
print(f"number of classes: {num_classes}\n")

print("loading model...")
try:
    model = load_model(model_path, num_classes, device)
except Exception as e:
    print(f"error loading model: {e}")
    print("\nTip: Make sure your checkpoint format matches the loading code.")
    exit(1)

print("Model loaded successfully!\n")

while True:
    print("=" * 60)
    image_path = input("enter image path ").strip()
    
    if image_path.lower() == 'q':
        print("Exiting...")
        break

    
    if not image_path:
        print("no file selected")
        break
    
    print(f"selected image: {image_path}\n")
    
    try:
        print("preprocessing image...")
        image_tensor = preprocess_image(image_path)
        
        print("running inference...\n")
        predicted_class, confidence, probabilities = predict(
            model, image_tensor, class_names, device
        )
        
        
        print("=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"predicted Class: {predicted_class}")
        print(f"confidence: {confidence:.2f}%")
        print("\nall class Probabilities:")
        print("-" * 60)
        
        
        probs_with_names = list(zip(class_names, probabilities.cpu().numpy()))
        probs_with_names.sort(key=lambda x: x[1], reverse=True)
        
        for class_name, prob in probs_with_names:
            bar = '█' * int(prob * 50)
            print(f"{class_name:.<20} {prob*100:>6.2f}% {bar}")
        
        print("=" * 60)
        print()
        
    except Exception as e:
        print(f"error dude {str(e)}")
        import traceback
        traceback.print_exc()
        print()
    
   