# Flower Classification with Deep Learning

This project implements a multi-class image classification system for flower images using deep learning techniques. The system includes three different models:
1. A custom CNN architecture
2. VGG16 as a feature extractor
3. Fine-tuned VGG16

## Project Structure

```
.
├── data/
│   ├── train/
│   └── test/ (optional)
├── results/           # Results, metrics, visualizations, and saved models are stored here
├── src/
│   ├── custom_cnn.py
│   ├── dataset.py
│   ├── main.py
│   ├── train.py
│   ├── utils.py
│   ├── vgg16_feature.py
│   ├── vgg16_finetune.py
│   └── visualize.py
├── requirements.txt
└── README.md
```
- The `results/` directory contains all metrics (`*_metrics.json`), training histories (`*_history.json`), visualizations (`*.png`), and saved models (`*.pth`).

## Requirements

- Python 3.8+
- torch
- torchvision
- numpy
- matplotlib
- scikit-learn
- tqdm
- Pillow
- (See requirements.txt for the full list)

To install dependencies:
```bash
pip install -r requirements.txt
```

## Training Procedure

1. Place the dataset in `data/train/` as described below.
2. Select the model: `custom_cnn`, `vgg16_feature`, or `vgg16_finetune`.
3. Set training parameters (epochs, batch size, learning rate).
4. Run the following command:
   ```bash
   python src/main.py --model [custom_cnn|vgg16_feature|vgg16_finetune] --epochs 20 --batch_size 32 --learning_rate 0.001
   ```
   - For Model 3 (VGG16 Fine-tuned), epochs will be set to 35 automatically.
5. All metrics, visualizations, and models are saved in the `results/` directory.

## How to Run

1. Set up the environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Place the dataset in `data/train/`.
3. Train a model:
   ```bash
   python src/main.py --model vgg16_finetune --epochs 35 --batch_size 32 --learning_rate 0.001
   ```
4. Results and visualizations will be generated in the `results/` directory.

## Dataset

The project uses the [Flowers Dataset](https://www.kaggle.com/datasets/imsparsh/flowers-dataset/data) from Kaggle, which contains five categories of flowers:
- Daisy
- Dandelion
- Rose
- Sunflower
- Tulip

**How to download and prepare the dataset:**
1. Register and log in to [Kaggle](https://www.kaggle.com/).
2. Go to the [Flowers Dataset page](https://www.kaggle.com/datasets/imsparsh/flowers-dataset/data).
3. Click "Download" and extract the contents.
4. Place the `train` folder inside the `data/` directory of this project. The structure should be:
   ```
   data/
   ├── train/
   │   ├── daisy/
   │   ├── dandelion/
   │   ├── rose/
   │   ├── sunflower/
   │   └── tulip/
   └── test/  # (optional, not used for evaluation)
   ```
5. Note: The dataset does not provide a labeled test set. All model evaluation and comparison is performed on a validation set automatically split from the training data.

## Model Architectures

### Custom CNN (Model 1)
- 5 convolutional blocks with batch normalization and max pooling
- 2 fully connected layers with dropout
- Input size: 224x224x3
- Output size: 5 (number of classes)
- Training time: ~99 minutes

### VGG16 Feature Extractor (Model 2)
- Uses pretrained VGG16 as feature extractor
- Freezes all VGG16 layers
- Adds custom classifier on top
- Input size: 224x224x3
- Output size: 5 (number of classes)
- Training time: ~71 minutes

### Fine-tuned VGG16 (Model 3)
- Uses pretrained VGG16
- Freezes first 2 convolutional blocks
- Fine-tunes remaining layers
- Enhanced classifier with:
  - Additional fully connected layer
  - Batch normalization
  - Increased dropout (0.6)
  - Weight decay (5e-4)
- Input size: 224x224x3
- Output size: 5 (number of classes)
- Training time: ~151 minutes

### Trained Model Files (Download Links)

You can download the trained weights for each model from the links below:

- [Custom CNN (Model 1)](https://drive.google.com/file/d/1pwxMXF-aGFcn3mUDfKefwnTyOdCQTlcQ/view?usp=sharing)
- [VGG16 Feature Extractor (Model 2)](https://drive.google.com/file/d/12AkCp_SdiMz5JjVk1KBJn45oopwysoE7/view?usp=sharing)
- [VGG16 Fine-Tuned (Model 3)](https://drive.google.com/file/d/1jWbGSi7_7Bd2ml6tvXaLNbHRjzyHpk_B/view?usp=sharing)
- [Best Performing Model (`best_model.pth` from Model 3)](https://drive.google.com/file/d/1ePmaNHpcxm9itJoY_BKc7TwigLJGBXFl/view?usp=sharing)

> Place the downloaded `.pth` files in the `results/` directory if you wish to reuse them for evaluation or visualization.

## Data Augmentation

The training data is augmented using:
- Random resized crop (224x224)
- Random horizontal flip
- Random rotation (±10 degrees)
- Color jittering (brightness, contrast, saturation)
- Normalization (ImageNet stats)

## Interpretation & Results

### Model Comparison Table

| Model                   | Accuracy | Precision | Recall  | F1-score | Train Time (s) |
|-------------------------|----------|-----------|---------|----------|----------------|
| Custom CNN              | 0.6685   | 0.6871    | 0.6685  | 0.6438   | 5942.89        |
| VGG16 Feature Extractor | 0.8871   | 0.8880    | 0.8871  | 0.8871   | 4266.76        |
| VGG16 Fine-tuned        | 0.9326   | 0.9339    | 0.9326  | 0.9325   | 9035.79        |

### Interpretation
- The VGG16 Fine-tuned model achieves the highest accuracy and F1-score, but takes the longest to train.
- The VGG16 Feature Extractor model is fast and accurate, demonstrating the power of transfer learning.
- The Custom CNN model performs worse than transfer learning models.
- There is a trade-off between training time and performance; fine-tuning yields the best results.

### Feature Visualization
- Feature maps from intermediate layers of each model are visualized in the `results/` directory.
- These visualizations help understand what features the models have learned.

### Reproducibility
- All code is modular and well-documented. Results can be reproduced by following the steps above.

## Credits

- This project was completed as part of the SE3508 Introduction to Artificial Intelligence course, instructed by Dr. Selim Yılmaz, Department of Software Engineering at Muğla Sıtkı Koçman University, 2025.

- Note: This repository must not be used by students in the same faculty in future years—whether partially or fully—as their own submission. Any form of code reuse without proper modification and original contribution will be considered by the instructor a violation of academic‐integrity policies.