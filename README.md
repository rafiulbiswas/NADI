# NADI
Multidialectal Arabic Speech Processing

# Arabic Dialect Identification using Enhanced ECAPA-TDNN

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-red.svg)
![SpeechBrain](https://img.shields.io/badge/SpeechBrain-v1.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Accuracy](https://img.shields.io/badge/accuracy-94.57%25-brightgreen.svg)

A high-performance Arabic dialect identification system achieving **94.57% accuracy** for the NADI 2025 shared task using enhanced ECAPA-TDNN architecture.

## 🎯 Key Features

- **94.57% Accuracy** on NADI 2025 validation set
- **8 Arabic Dialects** supported: Algeria, Egypt, Jordan, Mauritania, Morocco, Palestine, UAE, Yemen
- **Enhanced ECAPA-TDNN** with custom classification head
- **Pre-trained Model** available on [Hugging Face Hub](https://huggingface.co/rafiulbiswas/arabic-speech-dialect-identification)

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 94.57% |
| **Average Cost (LRE 2022)** | 0.067 |
| **Dialects** | 8 classes |
| **Dataset** | NADI 2025 (25,600 samples) |

## 🚀 Quick Start

### Using Pre-trained Model

```python
from speechbrain.inference.classifiers import EncoderClassifier

# Load the fine-tuned model
model = EncoderClassifier.from_hparams(
    source="rafiulbiswas/arabic-speech-dialect-identification",
    savedir="tmp"
)

# Predict dialect from audio file
prediction = model.classify_file("path/to/arabic_audio.wav")
print(f"Predicted dialect: {prediction}")
```

### Installation

```bash
pip install git+https://github.com/speechbrain/speechbrain.git@develop
pip install datasets==3.5.0 torch torchaudio transformers huggingface_hub
```

## 📂 Code & Notebooks

- **Subtask 1 (Dialect ID)**: `SpeechBrain_NADI_TASK1.ipynb` - Complete training pipeline
- **Subtask 2 (ASR)**: `SpeechBrain_NADI_TASK2.ipynb` - Speech recognition implementation

## 🎯 Dataset

### NADI 2025 Competition
- **Subtask 1**: [Spoken Language Identification](https://huggingface.co/datasets/UBC-NLP/NADI2025_subtask1_SLID)
- **Subtask 2**: [Automatic Speech Recognition](https://huggingface.co/datasets/UBC-NLP/NADI2025_subtask2_ASR)
- **Competition**: [NADI 2025 Shared Task](https://nadi.dlnlp.ai/2025/index.html#subtasks)

### Dataset Stats
- **Train**: 12,900 samples
- **Validation**: 12,700 samples  
- **Test**: 6,268 samples
- **Audio**: 16kHz, 1.04s-15.12s duration
- **Balance**: 3,200 samples per dialect

## 🏗️ Architecture

```
Input Audio → ECAPA-TDNN Encoder → Enhanced Classifier → 8 Dialect Classes
                    ↓
        [Frozen → Fine-tuned with Focal Loss + Augmentation]
```

## 🏆 Results

| Dialect | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Algeria | 0.95 | 0.94 | 0.94 |
| Egypt | 0.96 | 0.95 | 0.95 |
| Jordan | 0.93 | 0.94 | 0.94 |
| Mauritania | 0.94 | 0.95 | 0.95 |
| Morocco | 0.95 | 0.94 | 0.94 |
| Palestine | 0.94 | 0.95 | 0.94 |
| UAE | 0.95 | 0.94 | 0.95 |
| Yemen | 0.94 | 0.95 | 0.94 |



## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [SpeechBrain Team](https://speechbrain.github.io/) for the framework
- [UBC-NLP](https://huggingface.co/datasets/UBC-NLP/NADI2025_subtask1_SLID) for the dataset
- [NADI 2025 Organizers](https://nadi.dlnlp.ai/2025/index.html#subtasks) for the competition

---

⭐ **Star this repository if you find it helpful!** ⭐