# Multimodal Emotion Recognition using EEG, Thermal, and Digital Images

This repository contains the official implementation of our research work:

**“Multimodal Emotion Recognition via Fusion of EEG, Thermal Imaging, and Digital Facial Expressions”**  
Published in: 2025 11th International Conference on Communication and Signal Processing (ICCSP)  
DOI: [10.1109/ICCSP64183.2025.11088814](https://doi.org/10.1109/ICCSP64183.2025.11088814)

---

## 🎯 Objectives

- Examine EEG signal changes related to various emotional states and relate them to facial thermal and digital image features.
- Propose a deep-learning-based multimodal framework combining EEG, thermal, and digital image modalities for Facial Emotion Recognition (FER).
- Evaluate and compare the performance of multimodal fusion against single-modality approaches to assess improvements in classification accuracy.

---

## 🧠 Motivation

Traditional unimodal FER systems often fail in real-world conditions (lighting variations, occlusions, suppressed expressions). By integrating neural (EEG), physiological (thermal imaging), and visual (digital images) signals, we aim to build a robust multimodal FER model that is:

- More accurate under real-world conditions  
- More reliable for healthcare and affective computing  
- Applicable to human-computer interaction, security, and mental health monitoring  

---

## 📂 Dataset

**Participants:** 50 healthy volunteers (age 20–28, ethically approved by SRM IEC #2992/IEC/2021)  
**Stimuli:** 60-second audiovisual clips inducing six emotions: Happy, Sad, Neutral, Anger, Surprise, Fear  

**Modalities:**

| Modality | Description |
|----------|-------------|
| EEG | 10–20 electrode system, 100 Hz sampling, bandpass filtered 0.5–50 Hz |
| Thermal Images | FLIR A305SC infrared camera, 320×240 pixels |
| Digital Images | DSLR camera, high-resolution facial expressions |

**Hugging Face Dataset Link:**  
[Download full dataset](https://huggingface.co/datasets/bibekram/emotion_multimodal)  

> ⚡ Note: You can directly load the dataset in Python via Hugging Face `datasets` library:

```python
from datasets import load_dataset
dataset = load_dataset("bibekram/emotion_multimodal")
⚙️ Preprocessing

EEG:

Normalization

Bandpass filter (0.5–50 Hz)

Segmentation into 1–2s epochs

Features: Theta Power, Skewness, Entropy, Wavelet Coefficients

Thermal / Digital Images:

Cropping (face region)

Resizing (256×256 px)

Augmentation (rotation, flipping, noise)

Features: Entropy, Energy, ORB & AKAZE keypoints

🔗 Multimodal Fusion Framework

Flow:
EEG → EEG Features
Thermal → Thermal Features
Digital → Digital Features
All Features → Fusion → Deep Learning Classifier → Emotion Prediction

Fusion Strategies:

Feature-Level Fusion: Combine extracted features before classification

Decision-Level Fusion: Combine classifier outputs

Hybrid Fusion: Blend both

🛠️ Methodology

Feature Extraction

EEG: Power Spectral Density (PSD), wavelet decomposition

Thermal: Heat maps, texture analysis

Digital Images: Facial landmarks, CNN-based features

Deep Learning Methods

CNNs for spatial feature extraction from images

MLPs for EEG and combined features

LSTM / Transformer modules for capturing temporal dependencies (planned for future work)

Correlation Analysis

Strong modality correlations found, e.g.:

Surprise ↔ EEG Theta Power & Thermal Entropy (r = 0.605, p < 0.001)

Sadness ↔ EEG Skewness & Thermal Energy

Anger ↔ EEG Autocorrelation & AKAZE Keypoints

Classification

Algorithms tested: Decision Tree, k-NN, MLP

Evaluation Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

📊 Results

The proposed multimodal emotion recognition system was evaluated using EEG, thermal, and digital facial image data. Both single-modality models and multimodal fusion frameworks were tested, incorporating classical and deep learning approaches.

1. Single-Modality Performance
Modality	Deep Learning Model / Approach	Accuracy (%)	Precision	Recall	F1-Score
EEG	MLP (3 hidden layers: 512→256→128)	95	0.94	0.95	0.94
Thermal	CNN (ResNet-18, fine-tuned)	98.3	0.98	0.98	0.98
Digital	CNN (VGG-16, fine-tuned)	95	0.95	0.95	0.95

Insight: Thermal images achieved the highest single-modality accuracy, while EEG and digital images provided complementary information for subtle emotional cues.

2. Multimodal Fusion Performance

Three fusion strategies were implemented to combine EEG, thermal, and digital image features:

a) Feature-Level Fusion

Approach: Concatenate extracted features from all modalities.

Classifier: Deep MLP (3 hidden layers, ReLU activations, dropout 0.3).

Result: Accuracy 99%, significantly higher than single-modality performance.

b) Decision-Level Fusion

Approach: Fuse classifier outputs (probabilities) using weighted averaging.

Result: Accuracy ~98.7%, demonstrating robustness against noise or missing modality data.

c) Hybrid Fusion

Approach: Combine feature-level MLP outputs and decision-level CNN predictions.

Result: Best performance achieved:

Accuracy: 99%

Precision: 0.99

Recall: 0.99

F1-Score: 0.99

3. Deep Learning Architecture Details

CNNs for Images:

Thermal Images: ResNet-18 pretrained on ImageNet

Digital Images: VGG-16 pretrained on ImageNet

Fine-tuned on emotion-labeled datasets

EEG Processing:

Feature extraction: PSD, wavelets, entropy, skewness

MLP classifier for final emotion prediction

LSTM modules tested for temporal dependencies

Fusion MLP Network:

Input: Concatenated EEG + Thermal + Digital features

Hidden layers: 512 → 256 → 128 neurons, ReLU activations

Dropout: 0.3

Output: Softmax layer for six-class emotion classification

4. Correlation Analysis Across Modalities

Strong inter-modality correlations enhanced recognition:

Surprise: EEG Theta Power ↔ Thermal Entropy (r = 0.605, p < 0.001)

Sadness: EEG Skewness ↔ Thermal Energy

Anger: EEG Autocorrelation ↔ AKAZE Digital Features

Insight: Fusion leverages complementary strengths of neural, physiological, and visual cues for highly robust emotion recognition.

5. Key Takeaways

Multimodal fusion consistently outperforms single-modality models.

Deep learning models (CNNs + MLP) effectively capture spatial and temporal patterns.

Hybrid fusion provides robustness and high accuracy, suitable for real-world systems.

Achieves near-perfect performance (99% accuracy), validating the integration of EEG, thermal, and digital signals.

For detailed results, tables, plots, and confusion matrices, refer to the full project report

💡 Applications

Healthcare: Mental health tracking, depression detection

Human-Computer Interaction: Adaptive systems responding to emotions

Affective Computing: Emotion-aware AI systems

Security & Defense: Emotion-based surveillance and stress detection

🔮 Future Work

Real-time implementation on edge devices

Use of CNNs, LSTMs, and Transformers for temporal-spatial feature learning

Investigating early vs. late vs. hybrid fusion approaches

Extending dataset to cover cross-cultural emotion representation

📜 Citation

If you use this code or dataset in your research, please cite:

@INPROCEEDINGS{11088814,
  author={Bibek Ram, et al.},
  booktitle={2025 11th International Conference on Communication and Signal Processing (ICCSP)},
  title={Multimodal Emotion Recognition via Fusion of EEG, Thermal, and Digital Images},
  year={2025},
  pages={1-8},
  doi={10.1109/ICCSP64183.2025.11088814}
}
