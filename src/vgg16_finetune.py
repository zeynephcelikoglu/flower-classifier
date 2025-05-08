import torch
import torch.nn as nn
import torchvision.models as models

class VGG16FineTuned(nn.Module):
    def __init__(self, num_classes=5, freeze_blocks=2):
        super(VGG16FineTuned, self).__init__()
        
        # Load pretrained VGG16 model
        vgg16 = models.vgg16(pretrained=True)
        
        # Split features into convolutional blocks
        blocks = []
        block = []
        for layer in vgg16.features:
            block.append(layer)
            if isinstance(layer, nn.MaxPool2d):
                blocks.append(nn.Sequential(*block))
                block = []
        if block:
            blocks.append(nn.Sequential(*block))

        # Freeze first `freeze_blocks` blocks
        self.features = nn.Sequential(*blocks)
        for i, blk in enumerate(self.features):
            if i < freeze_blocks:
                for param in blk.parameters():
                    param.requires_grad = False
        
        # Modify the classifier with stronger regularization
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512),  # Add BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),  # Increased dropout
            nn.Linear(512, 256),  # Additional layer
            nn.BatchNorm1d(256),  # Add BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Additional dropout
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        x = self.features(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classify
        x = self.classifier(x)
        
        return x
    
    def get_features(self, x, layer_name):
        """
        Extract features from a specific layer for visualization.
        For VGG16, we use block1, block3, and block5 for visualization.
        """
        if not isinstance(layer_name, str) or not layer_name.startswith('block'):
            return None
            
        try:
            block_idx = int(layer_name[5:]) - 1  # Convert 'block1' to 0, 'block3' to 2, etc.
            if block_idx < 0 or block_idx >= len(self.features):
                return None
                
            # Extract features up to the requested block
            for i, block in enumerate(self.features):
                x = block(x)
                if i == block_idx:
                    return x
                    
        except (ValueError, IndexError):
            return None
            
        return None 