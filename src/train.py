import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
from vgg16_finetune import VGG16FineTuned

def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001, device='cuda', patience=5):
    """
    Train the model and return the trained model, training history, and total training time.
    Implements EarlyStopping based on validation loss.
    For VGG16 Fine-tuned model, uses different learning rates for conv and fc layers.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Special handling for VGG16 Fine-tuned model
    if isinstance(model, VGG16FineTuned):
        # Separate parameters into conv and fc layers
        conv_params = [p for p in model.features.parameters() if p.requires_grad]
        fc_params = model.classifier.parameters()
        
        # Create parameter groups with different learning rates and stronger weight decay
        optimizer = optim.Adam([
            {'params': conv_params, 'lr': learning_rate * 0.05},  # Even lower LR for conv layers
            {'params': fc_params, 'lr': learning_rate * 0.5}  # Lower LR for fc layers
        ], weight_decay=5e-4)  # Increased weight decay
        
        # Add learning rate scheduler for Model 3
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = None
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Step the scheduler for Model 3
        if scheduler is not None:
            scheduler.step(val_loss)  # Use validation loss for ReduceLROnPlateau
            # Print current learning rates
            current_lrs = [group['lr'] for group in optimizer.param_groups]
            print(f'Current learning rates: Conv={current_lrs[0]:.6f}, FC={current_lrs[1]:.6f}')
        
        # EarlyStopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement in val loss for {patience} epochs)")
                break
        
        # Save best model by accuracy as well (optional, for compatibility)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    total_time = time.time() - start_time
    return model, history, total_time

def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate the model on test data and return metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    return metrics 