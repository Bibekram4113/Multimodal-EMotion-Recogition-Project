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

Feature Extraction: EEG (PSD, wavelet), Thermal (heat maps, texture), Digital (landmarks, CNN features)

Correlation Analysis: Strong relationship found (e.g., EEG Theta Power ↔ Thermal Entropy for Surprise, r = 0.605, p < 0.001)

Classification: Decision Tree, k-NN, and MLP tested across modalities

Evaluation Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

📊 Results

Single-Modality Accuracy:

EEG: ~95%

Thermal: ~98.3%

Digital: ~95%

Multimodal Fusion: Improved accuracy (up to 99% in some classifiers)

Significant correlations:

Surprise ↔ Theta Power (EEG) & Entropy (Thermal)

Sadness ↔ EEG Skewness & Thermal Energy

Anger ↔ EEG Autocorrelation & AKAZE Keypoints

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
