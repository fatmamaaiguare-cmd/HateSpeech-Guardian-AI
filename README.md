# 🛡️ HateSpeech Guardian AI

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange.svg)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/Library-PyTorch-EE4C2C.svg)](https://pytorch.org/)

HateSpeech Guardian AI is an advanced NLP application designed to detect and highlight hate speech and toxic content at the **token level**. Instead of just classifying a whole sentence, it pinpoint specific words or phrases that trigger the toxicity.

## 🌟 Key Features
- **Token-Level Detection:** Highlights toxic words in red with a custom HTML/CSS interface.
- **Real-Time Analysis:** Adjust sensitivity thresholds in real-time using a slider.
- **Bulk Audit (CSV):** Upload large datasets to audit comments and export a toxicity report.
- **Visual Analytics:** Interactive charts powered by Plotly to visualize the distribution of safe vs. toxic content.
- **Context-Aware Logic:** Includes a custom algorithm to reduce false positives on common stop-words while maintaining high sensitivity for toxic neighbors.

## 🛠️ Tech Stack
- **Deep Learning Framework:** PyTorch & Hugging Face Transformers.
- **Model Architecture:** AutoModelForTokenClassification (Optimized Fine-tuned model).
- **Frontend:** Streamlit with custom CSS injection.
- **Data Handling:** Pandas & Plotly for analytics.

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9 or higher.
- A fine-tuned model folder (e.g., `final_hate_model_optimized`).

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone [https://github.com/fatmamaaiguare-cmd/HateSpeech-Guardian-AI.git](https://github.com/fatmamaaiguare-cmd/HateSpeech-Guardian-AI.git)
cd HateSpeech-Guardian-AI
pip install streamlit torch pandas plotly transformers
