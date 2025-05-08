import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import json
import os
import torch.nn as nn

def visualize_feature_maps(model, image_path, layer_name, device='cuda'):
    """
    Visualize feature maps from a specific layer of the model using get_features method.
    Works for both CustomCNN and VGG models.
    """
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get feature maps using model's get_features method
    model.eval()
    with torch.no_grad():
        feats = model.get_features(image_tensor, layer_name)
        if feats is None:
            raise ValueError(f"Layer {layer_name} not found in model's get_features method.")
        features = feats.squeeze().cpu().numpy()
    
    # Plot feature maps
    n_features = features.shape[0]
    n_cols = 8
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2.5 * n_rows))
    axes = axes.flatten()
    
    for i in range(n_features):
        if i < len(axes):
            axes[i].imshow(features[i], cmap='viridis')
            axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Feature Maps from {layer_name}', y=1.02)
    plt.tight_layout()
    
    return fig

def plot_training_history(history):
    """
    Plot training and validation metrics.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(confusion_matrix, class_names):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt.gcf()

class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_in, grad_out):
            # Only allow positive gradients to flow back
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
            return grad_in

        # Register hook for every ReLU
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                self.hooks.append(module.register_backward_hook(backward_hook))

    def generate_gradients(self, input_image, target_class):
        input_image = input_image.clone().detach().requires_grad_(True)
        output = self.model(input_image)
        self.model.zero_grad()
        grad_target_map = torch.zeros_like(output)
        grad_target_map[0][target_class] = 1
        output.backward(gradient=grad_target_map)
        gradients = input_image.grad.data[0].cpu().numpy()
        return gradients

    def close(self):
        for hook in self.hooks:
            hook.remove() 