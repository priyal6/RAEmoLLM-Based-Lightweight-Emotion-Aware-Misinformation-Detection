# RAEmoLLM: Lightweight Emotion-Aware Misinformation Detection

This project implements a lightweight version of the RAEmoLLM framework, integrating **emotion-aware embeddings** and **retrieval-augmented generation (RAG)** for effective misinformation detection across various domains. It supports in-context learning with TinyLLaMA and evaluates statements for truthfulness considering emotional context.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Folder Structure](#folder-structure)
- [Usage](#usage)
- [Scripts Overview](#scripts-overview)
- [Evaluation](#evaluation)
- [Reproducibility Notes](#reproducibility-notes)


## Features

- **Emotion-Aware Detection**: Integrates emotional context using DistilRoBERTa embeddings
- **Retrieval-Augmented Generation**: Uses ChromaDB for efficient similarity-based retrieval
- **Lightweight Architecture**: Optimized for resource-constrained environments using TinyLLaMA
- **Multi-Domain Support**: Works across various misinformation domains
- **Comprehensive Evaluation**: Includes accuracy, precision, recall, and F1-score metrics

## Installation

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.com/) installed on your system
- Git

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/priyal6/RAEmoLLM-Based-Lightweight-Emotion-Aware-Misinformation-Detection.git
   cd RAEmoLLM-Based-Lightweight-Emotion-Aware-Misinformation-Detection
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install TinyLLaMA via Ollama:**
   ```bash
   ollama pull tinylama
   ```

4. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Folder Structure

```
RAEmoLLM-Based-Lightweight-Emotion-Aware-Misinformation-Detection/
├── data/                          # Dataset files
├── models/                        # Trained model artifacts
├── scripts/                       # Core processing scripts
│   ├── quick_liar_setup.py       # Dataset preprocessing
│   ├── emotion_tagging.py        # Emotion-aware embedding generation
│   ├── build_vectorstore.py      # ChromaDB vector store creation
│   ├── construct_prompt.py       # RAG prompt construction
│   ├── ollama_inference.py       # TinyLLaMA inference
│   └── visualise_results.py      # Results evaluation and visualization
├── results/                       # Output files and visualizations
├── requirements.txt              # Python dependencies
├── main.py                       # Full pipeline execution
└── README.md                     # Project documentation
```

## Usage

### Quick Start - Full Pipeline

Run the complete RAEmoLLM pipeline:

```bash
python main.py
```

This executes all steps sequentially from preprocessing to visualization.

### Step-by-Step Execution

For more control over the process, run individual scripts:

1. **Preprocess the LIAR dataset:**
   ```bash
   python scripts/quick_liar_setup.py
   ```

2. **Generate emotion-aware embeddings:**
   ```bash
   python scripts/emotion_tagging.py
   ```

3. **Build the vector store:**
   ```bash
   python scripts/build_vectorstore.py
   ```

4. **Construct RAG prompts:**
   ```bash
   python scripts/construct_prompt.py
   ```

5. **Run inference:**
   ```bash
   python scripts/ollama_inference.py
   ```

6. **Evaluate and visualize results:**
   ```bash
   python scripts/visualise_results.py
   ```

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `quick_liar_setup.py` | Cleans and formats the LIAR dataset into a standard CSV for processing |
| `emotion_tagging.py` | Generates emotion-aware embeddings for each statement using DistilRoBERTa |
| `build_vectorstore.py` | Builds a ChromaDB vector store from the emotion-tagged dataset for retrieval-based inference |
| `construct_prompt.py` | Creates the RAEmoLLM in-context learning prompts using retrieved examples and emotional context |
| `ollama_inference.py` | Runs TinyLLaMA inference via Ollama on the prepared prompts |
| `visualise_results.py` | Computes evaluation metrics (accuracy, precision, recall, F1) and generates charts for analysis |
| `main.py` | Executes the full pipeline sequentially from preprocessing to visualization |

## Evaluation

The framework evaluates misinformation detection performance using standard metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

Results are automatically visualized with charts showing:
- Confusion matrices
- Performance metrics comparison
- Emotion distribution analysis

## Reproducibility Notes

To ensure reproducible results:

1. **Environment Consistency**: Use the exact Python version and dependencies specified in `requirements.txt`
2. **Random Seeds**: All scripts use fixed random seeds where applicable
3. **Model Versions**: Ensure you're using the same version of TinyLLaMA via Ollama
4. **Data Integrity**: The LIAR dataset preprocessing is deterministic

### System Requirements

- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: At least 5GB free space for models and data
- **CPU**: Multi-core processor recommended for faster processing

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this work in your research, please cite:

```bibtex
@software{raemollm2024,
  title={RAEmoLLM: Lightweight Emotion-Aware Misinformation Detection},
  author={Priyal Chugh},
  year={2024},
  url={https://github.com/priyal6/RAEmoLLM-Based-Lightweight-Emotion-Aware-Misinformation-Detection}
}
```


## Acknowledgments

- LIAR dataset creators for providing the benchmark dataset
- Hugging Face for DistilRoBERTa model
- Ollama team for the lightweight LLM inference framework
- ChromaDB for efficient vector storage and retrieval



---

**⭐ Star this repository if you find it helpful!**
