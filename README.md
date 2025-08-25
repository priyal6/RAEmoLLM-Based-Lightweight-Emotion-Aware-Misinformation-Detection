# RAEmoLLM-Based-Lightweight-Emotion-Aware-Misinformation-Detection
# RAEmoLLM: Lightweight Emotion-Aware Misinformation Detection

This project implements a lightweight version of the RAEmoLLM framework, integrating **emotion-aware embeddings** and **retrieval-augmented generation (RAG)** for effective misinformation detection across various domains. It supports in-context learning with TinyLLaMA and evaluates statements for truthfulness considering emotional context.

---

## Table of Contents
- [Installation](#installation)
- [Folder Structure](#folder-structure)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Reproducibility Notes](#reproducibility-notes)
- [License](#license)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/priyal6/RAEmoLLM-Based-Lightweight-Emotion-Aware-Misinformation-Detection.git
cd RAEmoLLM-Based-Lightweight-Emotion-Aware-Misinformation-Detection

---
## Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

Ensure Ollama is installed and TinyLLaMA is available locally:
ollama pull tinylama

Install dependencies:
pip install -r requirements.txt

Scripts Overview
Script	Purpose
quick_liar_setup.py	Cleans and formats the LIAR dataset into a standard CSV for processing.
emotion_tagging.py	Generates emotion-aware embeddings for each statement using DistilRoBERTa.
build_vectorstore.py	Builds a ChromaDB vector store from the emotion-tagged dataset for retrieval-based inference.
construct_prompt.py	Creates the RAEmoLLM in-context learning prompts using retrieved examples and emotional context.
ollama_inference.py	Runs TinyLLaMA inference via Ollama on the prepared prompts.
visualise_results.py	Computes evaluation metrics (accuracy, precision, recall, F1) and generates charts for analysis.
main.py	Executes the full pipeline sequentially from preprocessing to visualization.
