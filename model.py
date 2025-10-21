import torch
import torch.nn as nn

class myCNN(nn.Module):
    def __init__(self, num_classes=2): # two classes cuz were predecting between two brands
        super().__init__()
        self.features = nn.Sequential(
      
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(), #used gelu cuz whos using relu anymore lol
            nn.MaxPool2d(2, 2),
            
        
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
       
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            
         
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(2, 2), # enough blocks for such simple model
        )
        
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7)) # constant input dim
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512), #3d tensor to a 1dim vector
            nn.GELU(),
            nn.Dropout(0.5), # no to overfiting
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes) #final layer
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x