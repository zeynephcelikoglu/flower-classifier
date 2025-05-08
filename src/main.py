import argparse
import torch
import os
from dataset import get_data_loaders
from custom_cnn import CustomCNN
from vgg16_feature import VGG16FeatureExtractor
from vgg16_finetune import VGG16FineTuned
from train import train_model, evaluate_model
from utils import save_model, save_metrics, save_training_history, get_device
from visualize import visualize_feature_maps, plot_training_history
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Train flower classification models')
    parser.add_argument('--model', type=str, required=True, choices=['custom_cnn', 'vgg16_feature', 'vgg16_finetune'],
                      help='Model to train')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    args = parser.parse_args()

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Get data loaders (train and val only)
    train_loader, val_loader = get_data_loaders(
        args.data_dir,
        batch_size=args.batch_size
    )

    # Create model with specific parameters for Model 3
    if args.model == 'custom_cnn':
        model = CustomCNN()
    elif args.model == 'vgg16_feature':
        model = VGG16FeatureExtractor()
    else:  # vgg16_finetune
        model = VGG16FineTuned(freeze_blocks=2)  # Only Model 3 uses freeze_blocks=2
        if args.epochs == 20:  # Only Model 3 uses 35 epochs
            args.epochs = 35
            print(f"Model 3 (VGG16 Fine-tuned) using {args.epochs} epochs")

    # Train model
    model, history, train_time = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        patience=args.patience
    )

    # Evaluate model on validation set
    metrics = evaluate_model(model, val_loader, device=device)
    metrics['train_time'] = train_time

    # Save results
    os.makedirs('results', exist_ok=True)
    save_model(model, f'results/{args.model}.pth')
    save_metrics(metrics, f'results/{args.model}_metrics.json')
    save_training_history(history, f'results/{args.model}_history.json')

    # Plot training history
    fig = plot_training_history(history)
    plt.savefig(f'results/{args.model}_history.png')
    plt.close()

    # Visualize features for custom CNN and VGG16 fine-tuned
    if args.model in ['custom_cnn', 'vgg16_finetune']:
        # Get a sample image from validation set
        sample_image = next(iter(val_loader))[0][0]
        sample_image_path = f'results/sample_image.png'
        # Unnormalize for saving
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = sample_image.permute(1, 2, 0).numpy()
        img = (img * std) + mean
        img = np.clip(img, 0, 1)
        plt.imsave(sample_image_path, img)

        # Use correct layer names for visualization
        if args.model == 'custom_cnn':
            vis_layers = ['conv1', 'conv3', 'conv5']
        elif args.model == 'vgg16_finetune':
            vis_layers = ['block1', 'block3', 'block5']  # These correspond to VGG16 blocks
        else:
            vis_layers = []

        for layer in vis_layers:
            try:
                fig = visualize_feature_maps(model, sample_image_path, layer, device=device)
                plt.savefig(f'results/{args.model}_{layer}_features.png')
                plt.close()
            except Exception as e:
                print(f"Warning: Could not visualize layer {layer}: {str(e)}")

    print(f"\nResults saved in 'results' directory")
    print(f"Validation metrics: {metrics}")

if __name__ == '__main__':
    main() 