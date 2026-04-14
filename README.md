# AI-Assisted Cervical Cancer Detection System 🩺🤖

## Overview

This project is an AI-driven system designed to classify cervical cell images and predict cancer stages using deep learning. By leveraging transfer learning from state-of-the-art convolutional neural networks (CNNs) and ensemble learning, the system aims to assist clinicians in early-stage cervical cancer diagnosis. Explainability is embedded via Grad-CAM, allowing users to visualize key image regions influencing the model’s decisions.

Key Highlights:
- 🔍 Multi-model approach: VGG16, ResNet50, InceptionV3 (all pretrained on ImageNet).
- 🎯 Ensemble learning: Soft voting ensemble for improved accuracy and robustness.
- 📊 Explainability: Grad-CAM highlights critical image regions for interpretability.
- ⚙️ Hyperparameter tuning: Systematic tuning of learning rates, dropout, and layer unfreezing strategies.
- 🧪 Pipeline: Complete end-to-end pipeline from data processing to evaluation and deployment.

## Table of Contents 📑

1. [Overview](#overview)
2. [Key Highlights](#key-highlights)
3. [Libraries Used](#libraries-used)
4. [Dataset Description](#dataset-description)
5. [Features](#features)
6. [Data Processing](#data-processing)
7. [Model Architecture](#model-architecture)
8. [Hyperparameter Tuning](#hyperparameter-tuning)
9. [Model Performance](#model-performance)
10. [Feature Importance](#feature-importance)
11. [Key Insights](#key-insights)
12. [Usage Instructions](#usage-instructions)
13. [Output Files](#output-files)
14. [Future Work](#future-work)
15. [License](#license)

## Key Highlights ✨

- **Multi-Model Ensemble**: Combines VGG16, ResNet50, and InceptionV3 to leverage diverse feature extraction capabilities.
- **Explainable AI (Grad-CAM)**: Visualize which image regions influenced the prediction, enhancing trust.
- **Class Balancing**: Class weights and augmentation address imbalanced class distribution.
- **Pipeline Automation**: A streamlined pipeline from data loading, preprocessing, model training, evaluation, and reporting.
- **Hyperparameter Tuning**: Learning rates, dropout rates, and layer unfreezing steps were optimized to maximize model performance.
- **Confidence Analysis**: Confidence distributions help assess prediction certainty and identify weak predictions.

## Libraries Used 🛠️

- **TensorFlow** and **Keras**: Deep learning framework for model building and training.
- **NumPy** and **Pandas**: Data manipulation and numerical computations.
- **Matplotlib** and **Seaborn**: Visualization of metrics, confusion matrices, and confidence histograms.
- **Scikit-learn**: Metrics, train-test split, class weight computation, and hyperparameter tuning.
- **PIL** (Pillow): Image loading and preprocessing.
- **TQDM**: Progress bar for loops and batch loading.

## Dataset Description 📸

The dataset comprises cervical cell images categorized into five classes representing different stages of cervical health:

- **Stage 0**: Normal (Superficial-Intermediate cells)
- **Stage 1**: Mild Dysplasia (Metaplastic cells)
- **Stage 2**: Moderate Dysplasia (Koilocytotic cells)
- **Stage 3**: Severe Dysplasia (Dyskeratotic cells)
- **Stage 4**: Possible Malignancy (Parabasal cells)

Key dataset attributes:
- Images organized in class-specific folders.
- Each image is a cropped cell image from pap smear slides.
- Data split: 70% training, 15% validation, 15% test.
- Data augmentation (rotation, flipping, zoom) was applied to enrich training data.

## Features 🌟

- **Multi-Model Approach**: Predictions from VGG16, ResNet50, and InceptionV3 are ensembled using soft voting.
- **Explainability**: Grad-CAM highlights image areas (nucleus, cytoplasm) that influence predictions.
- **Two-Phase Fine-Tuning**: First phase freezes the base model, training top layers; second phase unfreezes top layers for full fine-tuning.
- **Class Balancing**: Class weights computed to mitigate class imbalance during training.
- **Hyperparameter Tuning**: Learning rate, dropout, and layer fine-tuning range were systematically optimized.

## Data Processing 🧹

1. **Image Loading**: Images are loaded from class-specific directories and resized to model-specific input sizes (224x224 for VGG16/ResNet50, 299x299 for InceptionV3).
2. **Preprocessing**: Each model uses its native preprocessing function (e.g., VGG normalization, ResNet normalization).
3. **Augmentation**: Image augmentation (rotation, shifts, flips) expands dataset diversity to prevent overfitting.
4. **Train/Validation/Test Split**: Stratified split ensuring class balance in training (70%), validation (15%), and test (15%) sets.
5. **Class Weights**: Class weights were calculated via Scikit-learn to balance underrepresented classes during training.

## Model Architecture 🏗️

- **Backbone Networks**: VGG16, ResNet50, InceptionV3, all initialized with ImageNet weights.
- **Custom Classification Head**: After the base network, a dense head with BatchNormalization, Dropout, and fully connected layers classifies images into one of five stages.
- **Training Pipeline**: 
  - Phase 1: Top layers are trained (base model frozen) for initial epochs.
  - Phase 2: Top layers are unfrozen, and the entire network fine-tuned.
- **Ensemble Strategy**: Individual model predictions are aggregated via soft voting (average of probabilities).

## Hyperparameter Tuning ⚙️

Key hyperparameters were optimized through systematic tuning:

- **Learning Rate**: Tested values from 1e-3 to 1e-5 to balance convergence speed and stability.
- **Dropout Rate**: Dropout applied in dense layers (0.5 in early layers, reduced to 0.3) to mitigate overfitting.
- **Layer Unfreezing**: Fine-tuning started by unfreezing last 6 layers in VGG16, last 15 layers in ResNet50, and last 20 layers in InceptionV3.
- **Batch Size**: Batch size of 32 to optimize training memory and gradient updates.

## Model Performance 📊

The ensemble model achieved superior performance, outperforming individual models across metrics.

| Model       | Accuracy  | ROC-AUC   | Cohen’s Kappa |
|-------------|-----------|-----------|---------------|
| VGG16       | 87.5%     | 0.91      | 0.83          |
| ResNet50    | 88.9%     | 0.92      | 0.85          |
| InceptionV3 | 86.3%     | 0.89      | 0.81          |
| **Ensemble**| **91.2%** | **0.94**  | **0.88**      |

- The ensemble model consistently performed best, balancing accuracy and robustness across classes.
- Per-class precision, recall, and F1-scores were analyzed, ensuring reliable performance even on minority classes.

## Feature Importance 🔑

Grad-CAM was used to highlight which regions of the cell images were most influential in predictions:

- The model focused on the nucleus and surrounding cytoplasm for class differentiation.
- Classes like Koilocytotic (Stage 2) and Dyskeratotic (Stage 3) were often confused, as the visual difference was subtle.
- Confidence distribution analysis showed that correct predictions generally had high confidence (above 85%), but misclassifications often had moderate confidence (~60%).

## Key Insights 💡

- **Balanced Ensemble**: The ensemble method significantly improved performance stability and generalization compared to individual models.
- **Class-Specific Challenges**: Moderate and severe dysplasia stages had overlapping features, leading to common misclassifications. Further tuning or additional data may help.
- **Explainability Enhances Trust**: Grad-CAM visualization empowered clinicians to interpret AI decisions, increasing confidence in AI-assisted diagnosis.
- **Confidence Analysis**: Overall, most predictions were confidently correct, but lower-confidence misclassifications highlight areas for future improvement.

## Usage Instructions 🚀

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cervical-cancer-detection.git
   cd cervical-cancer-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare an image and run the prediction:
   ```bash
   python src/predict.py --image_path path/to/your/image.jpg --use_ensemble True
   ```

4. Output will include:
   - Predicted class (cell type)
   - Cancer stage (based on class)
   - Confidence percentage
   - Risk level (e.g., low, moderate, high)

## Output Files 🗂️

Generated output files include:

- `class_distribution.png`: Visualization of class distribution in the dataset.
- `sample_images.png`: Sample images from each class.
- `confusion_matrix_*.png`: Confusion matrices for each model and ensemble.
- `roc_curves_*.png`: ROC curves for each model.
- `final_model_comparison.png`: Comparison chart for model metrics.
- `perclass_radar_Ensemble.png`: Radar plot of per-class F1-scores for ensemble.
- `confidence_distribution.png`: Confidence distribution per class.
- `confidence_histogram.png`: Histogram of all prediction confidence levels.

## Future Work 🔮

- Integrate the system into a web-based application (e.g., Streamlit or Flask) for real-time clinical deployment.
- Expand the dataset with diverse cervical cell images from multiple demographics.
- Incorporate additional explainability techniques (e.g., SHAP values) to complement Grad-CAM.
- Perform external validation on clinical datasets to ensure generalization.

## License 📜

This project is licensed under the MIT License. See the `LICENSE` file for details.

