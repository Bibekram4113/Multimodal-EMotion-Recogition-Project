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

⚡ Dataset Loading

You can load the multimodal dataset directly using Hugging Face:

from datasets import load_dataset
dataset = load_dataset("bibekram/emotion_multimodal")

⚙️ Preprocessing

EEG:

Normalization

Bandpass filter (0.5–50 Hz)

Segmentation into 1–2s epochs

Feature extraction: Theta Power, Skewness, Entropy, Wavelet Coefficients

Thermal / Digital Images:

Face cropping

Resizing to 256×256 px

Data augmentation: rotation, flipping, noise

Feature extraction: Entropy, Energy, ORB & AKAZE keypoints

🔗 Multimodal Fusion Framework

Flow:

EEG → EEG Features
Thermal → Thermal Features
Digital → Digital Features
All Features → Fusion → Deep Learning Classifier → Emotion Prediction


Fusion Strategies:

Feature-Level Fusion: Combine extracted features before classification

Decision-Level Fusion: Combine classifier outputs

Hybrid Fusion: Blend both approaches

🛠️ Methodology

Feature Extraction:

EEG: Power Spectral Density (PSD), wavelet decomposition

Thermal: Heat maps, texture analysis

Digital: Facial landmarks, CNN-based features

Deep Learning Models:

CNNs for spatial feature extraction from images

MLPs for EEG and fused features

LSTM / Transformer modules for temporal dependencies (planned for future work)

Correlation Analysis:

Surprise ↔ EEG Theta Power & Thermal Entropy (r = 0.605, p < 0.001)

Sadness ↔ EEG Skewness & Thermal Energy

Anger ↔ EEG Autocorrelation & AKAZE Keypoints

Classification:

Algorithms tested: Decision Tree, k-NN, MLP

Evaluation Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

📊 Results
Single-Modality Performance
Modality	Model / Approach	Accuracy (%)	Precision	Recall	F1-Score
EEG	MLP (512→256→128)	95	0.94	0.95	0.94
Thermal	CNN (ResNet-18, fine-tuned)	98.3	0.98	0.98	0.98
Digital	CNN (VGG-16, fine-tuned)	95	0.95	0.95	0.95

Insight: Thermal images achieved the highest single-modality accuracy, while EEG and digital images provide complementary information for subtle emotional cues.

Multimodal Fusion Performance

Feature-Level Fusion: Concatenated features → MLP → Accuracy 99%

Decision-Level Fusion: Weighted averaging of classifier outputs → Accuracy ~98.7%

Hybrid Fusion: Combines feature- and decision-level outputs → Accuracy 99%, Precision 0.99, Recall 0.99, F1-Score 0.99

Deep Learning Architecture:

CNNs (ResNet-18 for Thermal, VGG-16 for Digital) pretrained on ImageNet and fine-tuned

EEG features → MLP classifier (optional LSTM for temporal patterns)

Fusion MLP: 3 hidden layers (512 → 256 → 128), ReLU, dropout 0.3, softmax output

Correlation Across Modalities:

Strong inter-modality correlations improve recognition:

Surprise: EEG Theta Power ↔ Thermal Entropy

Sadness: EEG Skewness ↔ Thermal Energy

Anger: EEG Autocorrelation ↔ AKAZE Digital Features

For detailed results, tables, plots, and confusion matrices, refer to the full project report
.

💡 Applications

Mental health monitoring and depression detection

Adaptive human-computer interaction systems

Emotion-aware AI applications

Security and surveillance systems

🔮 Future Work

Real-time implementation on edge devices

Advanced deep learning models: CNN-LSTM, Transformers

Investigating early, late, and hybrid fusion strategies

Extending dataset for cross-cultural emotion recognition

📜 Citation

ICCSP Paper:

@INPROCEEDINGS{11088814,
  author={Bibek Ram, et al.},
  booktitle={2025 11th International Conference on Communication and Signal Processing (ICCSP)},
  title={Multimodal Emotion Recognition via Fusion of EEG, Thermal, and Digital Images},
  year={2025},
  pages={1-8},
  doi={10.1109/ICCSP64183.2025.11088814}
}


Project Report:

@MISC{bibek2025projectreport,
  author = {Bibek Ram},
  title = {Multimodal Emotion Recognition via Fusion of EEG, Thermal, and Digital Images – Project Report},
  year = {2025},
  howpublished = {Project Report, SRM Institute of Science and Technology},
  note = {Available at: Project Report/project_report.docx}
}
