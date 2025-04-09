# gan-enhanced-pneumonia-classifier

Enhancing Pneumonia Detection with GAN-Generated Synthetic Chest X-rays

## Table of Contents
- [Task](#1-task)
- [Related Work](#2-related-work)
- [Approach](#3-approach)
- [Dataset and Metrics](#4-dataset-and-metrics)
- [File Structure](#5-file-structure)
- [Setup and Usage](#6-setup-and-usage)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Dataset Preparation](#dataset-preparation)
  - [Training the Baseline Classifier](#training-the-baseline-classifier)
  - [Training the GAN](#training-the-gan)
  - [Generating Synthetic Images](#generating-synthetic-images)
  - [Training the Augmented Classifier](#training-the-augmented-classifier)
  - [Analyzing Results](#analyzing-results)
- [References](#7-references)

## 1. Task
This project seeks to create realistic synthetic X-ray images of lungs with pneumonia using a Generative Adversarial Network (GAN) and assess their effect on enhancing a deep learning classifier's performance. By enriching the dataset with high-quality synthetic images, we aim to boost model accuracy and generalization, especially in scenarios with scarce medical data. The difficulty is in ensuring these images maintain authentic anatomical features and pneumonia-specific traits, like lung opacities, while preventing artifacts that might confuse the classifier. The synthetic images must faithfully represent pneumonia signs without adding deceptive patterns that could harm classifier effectiveness.  

## 2. Related Work
Generative Adversarial Networks (GANs) have progressed synthetic image creation in medical imaging. Goodfellow et al. (2014) pioneered GANs, achieving lifelike image generation, though initial versions grappled with instability and poor quality [1]. Frid-Adar et al. (2018) employed GANs to enhance liver lesion datasets, boosting CNN accuracy, yet struggled with realism for intricate anatomies [2]. Yi et al. (2019) surveyed GANs in medical contexts, citing X-ray augmentation wins but underscoring problems with retaining diagnostic traits [3]. Kazeminia et al. (2020) evaluated GANs for medical image tasks, stressing X-ray synthesis promise while noting challenges in pathology-specific detail capture [4]. These works expose flaws in anatomical accuracy and feature retention—gaps we tackle with conditional generation and improved loss functions.

## 3. Approach
We will implement a Deep Convolutional GAN (DCGAN) using PyTorch to generate synthetic chest X-rays, conditioned on pneumonia labels to ensure relevant feature synthesis. To improve realism, we'll augment the standard adversarial loss with a perceptual loss using a pre-trained VGG-16 network, encouraging anatomical detail preservation. We'll adapt the open-source DCGAN implementation from PyTorch's examples, adding custom code for label conditioning and perceptual loss. Synthetic images will be generated (targeting 5,000 additional samples), processed (resized to 224x224, normalized), and combined with the original training set. A pre-trained ResNet-50 classifier will then be fine-tuned on this augmented dataset using cross-entropy loss, with performance compared against a baseline trained on the original data alone. 

## 4. Dataset and Metrics
The project utilizes the RSNA Pneumonia Detection Challenge dataset from Kaggle ([link](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)), specifically using a pre-processed version available at ([link](https://www.kaggle.com/datasets/iamtapendu/rsna-pneumonia-processed-dataset)). This version includes metadata files (`stage2_train_metadata.csv`, `stage2_test_metadata.csv`) and corresponding PNG images located in `Training/Images/` and `Test/` directories within the download. The training metadata contains class labels ("Normal," "No Lung Opacity/Not Normal," "Lung Opacity"), while the test metadata uses a "PredictionString" format. The dataset comprises approximately 26,684 training images and 6,671 test images (actual numbers may vary slightly based on the metadata). Our `data_loader.py` script processes this structure, converting it into a binary classification task: "Lung Opacity" (label 1) vs. all others (label 0). Minimal pre-processing is applied using standard ImageNet normalization and resizing to 224x224 via `torchvision.transforms`. Our primary metric is classification accuracy, aiming for over 85% on the test set, compared against a baseline ResNet-50's performance (initially around 80%). We also track weighted precision, recall, and F1-score. Validation is performed using 5-fold cross-validation by default, configurable via script arguments.

## 5. File Structure
```
gan-enhanced-pneumonia-classifier/
│
├── .gitignore           # Git ignore rules
├── .gitattributes      # Git attributes configuration
├── LICENSE             # Project license
├── README.md           # Project documentation
├── requirements.txt    # Project dependencies
│
├── data/
│   ├── processed/      # Location for the downloaded and extracted dataset
│   │   ├── stage2_train_metadata.csv
│   │   ├── stage2_test_metadata.csv
│   │   ├── Training/
│   │   │   └── Images/ # Training PNG images
│   │   └── Test/       # Test PNG images
│   └── synthetic/      # Generated synthetic images
│
├── models/
│   ├── baseline_resnet50.pth      # Baseline classifier checkpoint
│   ├── augmented_resnet50.pth     # Augmented classifier checkpoint
│   └── gan/                       # GAN model checkpoints
│       ├── generator_final.pth
│       └── discriminator_final.pth
│
├── notebooks/          # Jupyter notebooks for exploration and visualization
│
├── results/
│   ├── figures/        # Training curves and performance plots
│   │   ├── baseline_*.png
│   │   └── augmented_*.png
│   ├── metrics/        # Training and evaluation metrics
│   │   ├── training_history.json
│   │   ├── final_evaluation_metrics.json
│   │   ├── cv_summary_metrics.json
│   │   └── gan_training_history.json
│   ├── analysis/       # Analysis outputs
│   │   ├── comparison_*.png       # Comparative visualizations
│   │   └── comparison_report.txt  # Detailed performance report
│   └── gan_images/     # GAN-generated sample images during training
│
├── src/
│   ├── __init__.py
│   ├── analyze_results.py    # Results analysis and visualization
│   ├── classifier.py         # ResNet50 model implementation
│   ├── data_loader.py       # Dataset loading and preprocessing
│   ├── dcgan.py             # DCGAN architecture implementation
│   ├── download_dataset.py   # Dataset download script
│   ├── generate_synthetic.py # Synthetic image generation
│   ├── train_classifier.py   # Classifier training script
│   ├── train_gan.py         # GAN training script
│   └── utils.py             # Utility functions
│
└── tests/              # Unit tests for source code
    └── __init__.py

```

The project follows a modular structure:

- **Source Code (`src/`)**: Core implementation files
  - Model architectures (`classifier.py`, `dcgan.py`)
  - Training scripts (`train_classifier.py`, `train_gan.py`)
  - Data handling (`data_loader.py`, `download_dataset.py`)
  - Analysis tools (`analyze_results.py`)

- **Data Management (`data/`)**: 
  - Original dataset in `processed/`
  - Synthetic images in `synthetic/`

- **Results (`results/`)**:
  - Training metrics and evaluation results
  - Performance visualizations
  - Analysis reports and comparisons
  - Generated samples from GAN

- **Models (`models/`)**: 
  - Saved model checkpoints
  - Both baseline and augmented versions

Each component is designed to be modular and reusable, with clear separation of concerns between data processing, model implementation, training, and analysis.

## 6. Setup and Usage

### Prerequisites

- Python 3.7+
- PyTorch 1.9+
- CUDA-compatible GPU (optional but recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/gan-enhanced-pneumonia-classifier.git
   cd gan-enhanced-pneumonia-classifier
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Windows
   py -m venv .venv
   .\.venv\Scripts\activate

   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch with CUDA support**
   ```bash
   pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
   ```

5. **Set up Kaggle API**
   - Create a Kaggle account at [Kaggle](https://www.kaggle.com/)
   - Go to your account settings
   - Create new API token
   - Set up credentials:
     ```bash
     mkdir -p ~/.kaggle
     cp /path/to/downloaded/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```

### Dataset Preparation

1. **Download the dataset**
   ```bash
   python src/download_dataset.py
   ```

2. **Verify dataset structure**
   ```bash
   python src/data_loader.py
   ```

   Expected structure:
   ```
   ./data/processed/
   ├── stage2_train_metadata.csv
   ├── stage2_test_metadata.csv
   ├── Training/
   │   └── Images/
   └── Test/
   ```

### Training the Baseline Classifier

**Basic training:**
```bash
python src/train_classifier.py
```

**Available options:**
```bash
--data-dir PATH      # Dataset directory (default: ./data/processed)
--model-dir PATH     # Model save directory (default: ./models)
--results-dir PATH   # Results directory (default: ./results/metrics)
--figures-dir PATH   # Figures directory (default: ./results/figures)
--epochs N          # Training epochs (default: 15)
--batch-size N      # Batch size (default: 32)
--lr FLOAT         # Learning rate (default: 0.001)
--unfreeze         # Unfreeze base ResNet layers
--k-folds N        # Cross-validation folds (default: 5)
--workers N        # Data loading workers (default: 4)
--cpu              # Force CPU usage
```

### Training the GAN

**Basic training:**
```bash
python src/train_gan.py
```

**Available options:**
```bash
--data-dir PATH     # Dataset directory
--model-dir PATH    # Model save directory
--output-dir PATH   # Base results directory
--results-dir PATH  # Metrics directory
--figures-dir PATH  # Figures directory
--epochs N         # Training epochs (default: 50)
--batch-size N     # Training batch size (default: 128)
--lr FLOAT        # Learning rate (default: 0.0002)
--latent-dim N    # Latent vector size (default: 100)
```

### Generating Synthetic Images

1. **Generate images using trained GAN:**
   ```bash
   python src/generate_synthetic.py --model-path ./models/gan/generator_final.pth
   ```

2. **Available options:**
   ```bash
   --model-path PATH  # Path to generator model (required)
   --output-dir PATH  # Output directory (default: ./data/synthetic)
   --num-images N     # Number of images to generate (default: 5000)
   --batch-size N     # Generation batch size (default: 64)
   ```

### Training the Augmented Classifier

1. **Train with synthetic data:**
   ```bash
   python src/train_classifier.py --use-synthetic
   ```

2. **Advanced configuration:**
   ```bash
   python src/train_classifier.py \
       --use-synthetic \
       --synthetic-dir /path/to/synthetic/images \
       --epochs 20 \
       --batch-size 64 \
       --lr 0.0005 \
       --unfreeze \
       --k-folds 5
   ```

### Analyzing Results

1. **Run analysis script:**
   ```bash
   python src/analyze_results.py
   ```

2. **Available options:**
   ```bash
   --metrics-dir PATH  # Metrics directory (default: ./results/metrics)
   --figures-dir PATH  # Output directory (default: ./results/analysis)
   ```

3. **Generated outputs:**
   - Comparative visualizations (`./results/analysis/comparison_*.png`)
   - Performance report (`./results/analysis/comparison_report.txt`)
   - Cross-validation analysis
   - Training curves comparison

## 7. References
[1] Goodfellow, I., et al. (2014). Generative Adversarial Nets. *Advances in Neural Information Processing Systems*.

[2] Frid-Adar, M., et al. (2018). GAN-based synthetic medical image augmentation for increased CNN performance in liver lesion classification. *Neurocomputing*.

[3] Yi, X., et al. (2019). Generative Adversarial Network in Medical Imaging: A Review. *Medical Image Analysis*.

[4] Kazeminia, S., et al. (2020). GANs for Medical Image Analysis. *arXiv preprint arXiv:2006.01668*.


 
