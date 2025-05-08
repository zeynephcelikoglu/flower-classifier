import torch
import torch.nn as nn
import torchvision.models as models

class VGG16FeatureExtractor(nn.Module):
    def __init__(self, num_classes=5):
        super(VGG16FeatureExtractor, self).__init__()
        
        # Load pretrained VGG16 model
        vgg16 = models.vgg16(pretrained=True)
        
        # Remove the last fully connected layer
        self.features = vgg16.features
        
        # Freeze all layers
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Add new classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
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
        Extract features from a specific layer for visualization
        """
        features = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                features[f'conv{i+1}'] = x
        return features.get(layer_name, None) 