import os
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import json
import matplotlib.pyplot as plt

def get_device():
    """
    Get the device to use for training (GPU if available, else CPU).
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def save_model(model, path):
    """
    Save model state dictionary.
    """
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """
    Load model state dictionary.
    """
    model.load_state_dict(torch.load(path))
    return model

def save_metrics(metrics, path):
    """
    Save evaluation metrics to a JSON file.
    """
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)

def load_metrics(path):
    """
    Load evaluation metrics from a JSON file.
    """
    with open(path, 'r') as f:
        return json.load(f)

def save_training_history(history, path):
    """
    Save training history to a JSON file.
    """
    with open(path, 'w') as f:
        json.dump(history, f, indent=4)

def load_training_history(path):
    """
    Load training history from a JSON file.
    """
    with open(path, 'r') as f:
        return json.load(f)

def create_experiment_dir(base_dir='experiments'):
    """
    Create a new experiment directory with a unique name.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Get the next experiment number
    existing_experiments = [d for d in os.listdir(base_dir) if d.startswith('exp_')]
    if existing_experiments:
        last_exp = max([int(d.split('_')[1]) for d in existing_experiments])
        exp_num = last_exp + 1
    else:
        exp_num = 1
    
    # Create new experiment directory
    exp_dir = os.path.join(base_dir, f'exp_{exp_num}')
    os.makedirs(exp_dir)
    
    return exp_dir

def create_results_table(models_metrics):
    """
    Create a table comparing model metrics
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Hide axes
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = []
    headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Train Time (s)']
    
    for model_name, metrics in models_metrics.items():
        row = [
            model_name,
            f"{metrics['accuracy']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1_score']:.4f}",
            f"{metrics['train_time']:.2f}"
        ]
        table_data.append(row)
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center'
    )
    
    # Adjust table style
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    plt.title('Model Comparison')
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