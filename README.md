# 🔮 Telugu-English Code-Mixed Next Word Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)

**A next-word prediction system for Telugu-English code-mixed text using Statistical and Neural Language Models**

[Features](#-features) • [Demo](#-quick-demo) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Models](#-models)

</div>

---

## 🎯 Overview

This project implements **4 different language models** for predicting the next word in Telugu-English code-mixed (Tenglish) text — a common style of communication among Telugu speakers.

```
Input:  "nenu eppudu"  (I always...)
Output: "untanu" ✓     (I will always be...)
```

### What is Code-Mixing?
Code-mixing is the practice of alternating between two or more languages in a single conversation. For example:
> *"Nenu tomorrow office ki veltanu"* (I will go to office tomorrow)

This project builds models that understand and predict such mixed-language patterns.

---

## ✨ Features

- 🧠 **4 Language Models**: N-gram, HMM, Hybrid, and LSTM
- 🔧 **Pure Python LSTM**: Built from scratch with NumPy (no PyTorch/TensorFlow)
- 📊 **Complete Preprocessing Pipeline**: Tokenization, normalization, lemmatization
- 🎮 **Interactive Demo**: Test predictions in real-time
- 📈 **Comprehensive Evaluation**: Top-1, Top-5 accuracy, and MRR metrics

---

## 🚀 Quick Demo

```bash
# Clone the repository
git clone https://github.com/mohanganesh3/telugu-english-codemix-nextword-predictor.git
cd telugu-english-codemix-nextword-predictor

# Install dependencies
pip install numpy

# Run interactive demo
python interactive_demo.py
```

**Example Output:**
```
Enter text: nenu school ki

📊 Predictions:
┌─────────────┬────────────────────────────────┐
│ Model       │ Top Predictions                │
├─────────────┼────────────────────────────────┤
│ N-gram      │ veltanu, velta, veldam         │
│ HMM         │ velta, veltanu, potha          │
│ Hybrid      │ veltanu, velta, veldam         │
│ LSTM        │ veltanu, oka, untanu           │
└─────────────┴────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites
- Python 3.7 or higher
- pip

### Setup

```bash
# Clone repository
git clone https://github.com/mohanganesh3/telugu-english-codemix-nextword-predictor.git
cd telugu-english-codemix-nextword-predictor

# Install dependencies
pip install numpy

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install numpy
```

---

## 💻 Usage

### 1. Interactive Testing (Recommended)
```bash
python interactive_demo.py
```

### 2. Train All Models
```bash
python train_all_models.py
```

### 3. Evaluate Models
```bash
python run_final_evaluation.py
```

### 4. Use in Your Code
```python
from models.hybrid.hybrid_language_tagger import HybridLanguageTagger

# Load the best model (Hybrid)
model = HybridLanguageTagger()
model.load_models('models/hybrid_tagger.txt')

# Predict next word
context = ['nenu', 'eppudu']
predictions = model.predict_next_word(context, top_k=5)
print(predictions)
# Output: [('untanu', 0.51), ('unte', 0.28), ('avutanu', 0.11), ...]
```

---

## 📊 Results

### Model Comparison

| Model | Top-1 Accuracy | Top-5 Accuracy | MRR |
|:------|:-------------:|:--------------:|:---:|
| N-gram (10-gram) | 7.21% | 11.67% | 0.089 |
| HMM (4-state) | 4.72% | 10.27% | 0.067 |
| **Hybrid** ⭐ | **7.33%** | **12.40%** | **0.092** |
| LSTM (2 epochs) | 0.00% | 0.00% | 0.000 |

> 🏆 **Best Model**: Hybrid (N-gram + HMM combined)

### Why These Accuracy Numbers?
- Code-mixed text is inherently unpredictable
- Limited dataset (~2,400 sentences)
- LSTM trained for only 2 epochs (each epoch = 24 hours on CPU!)
- Pure Python implementation for educational purposes

---

## 🧠 Models

### 1. N-gram Model
- **Type**: Statistical
- **N**: 10-gram with backoff
- **Smoothing**: Kneser-Ney
- **Speed**: ⚡ Fast

### 2. Hidden Markov Model (HMM)
- **Type**: Statistical
- **States**: 4 hidden states
- **Smoothing**: Laplace
- **Training**: Baum-Welch algorithm

### 3. Hybrid Model ⭐
- **Type**: Ensemble
- **Approach**: Combines N-gram + HMM
- **Weighting**: Adaptive confidence-based

### 4. LSTM Model
- **Type**: Neural Network
- **Architecture**: 2-layer LSTM
- **Hidden Units**: 256
- **Embedding**: 128 dimensions
- **Implementation**: Pure Python/NumPy (educational purpose)

---

## 📁 Project Structure

```
telugu-english-codemix-nextword-predictor/
├── 📂 data/
│   ├── processed/          # Preprocessed train/val/test data
│   └── vocabulary.txt      # Vocabulary file
├── 📂 src/
│   ├── preprocessing/      # Tokenizer, normalizer, lemmatizer
│   └── utils/              # Configuration utilities
├── 📂 models/
│   ├── ngram/              # N-gram model
│   ├── hmm/                # HMM model
│   ├── hybrid/             # Hybrid model
│   └── lstm/               # LSTM model
├── 🎮 interactive_demo.py  # Interactive testing
├── 🏋️ train_all_models.py  # Train all models
├── 📊 run_final_evaluation.py
└── 📖 README.md
```

---

## 📚 Dataset

- **Size**: ~2,400 Telugu-English code-mixed sentences
- **Split**: 70% train, 15% validation, 15% test
- **Domain**: Casual conversations, social media style

---

## 🔮 Future Improvements

- [ ] Train LSTM with PyTorch/TensorFlow (GPU acceleration)
- [ ] Implement Transformer architecture
- [ ] Add attention mechanism
- [ ] Expand dataset to 10,000+ sentences
- [ ] Build web interface for demo

---

## 🛠️ Tech Stack

- **Language**: Python 3.7+
- **Core Library**: NumPy
- **Models**: Statistical (N-gram, HMM) + Neural (LSTM)
- **No frameworks**: Pure Python implementation for educational purposes

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- IIIT Sri City, Chittoor
- NLP Course Faculty

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ by [Mohan Ganesh](https://github.com/mohanganesh3)

</div>
