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
Create and activate a virtual environment:

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Ensure Ollama is installed and TinyLLaMA is available locally:

bash
Copy
Edit
ollama pull tinylama
Folder Structure
plaintext
Copy
Edit
RAEmoLLM/
├── data/
│   └── liar_dataset.csv           # LIAR dataset
├── models/
│   └── tinyllama/                 # TinyLLaMA model files
├── scripts/
│   ├── quick_liar_setup.py        # Dataset preprocessing
│   ├── emotion_tagging.py         # Emotion embedding extraction
│   ├── build_vectorstore.py       # Build ChromaDB vector store
│   ├── construct_prompt.py        # Prompt construction for RAEmoLLM
│   ├── ollama_inference.py        # Run TinyLLaMA inference
│   └── visualise_results.py       # Visualization and evaluation charts
├── results/
│   ├── baseline_metrics.json
│   ├── raemollm_metrics.json
│   └── charts/
├── requirements.txt
├── main.py                        # End-to-end execution
└── README.md
Usage
Run the pipeline step by step:

1. Prepare Dataset

bash
Copy
Edit
python scripts/quick_liar_setup.py
2. Emotion Tagging

bash
Copy
Edit
python scripts/emotion_tagging.py --input data/processed_misinfo.csv --output data/emotion_tagged_misinfo.csv
3. Build Vector Store

bash
Copy
Edit
python scripts/build_vectorstore.py --input data/emotion_tagged_misinfo.csv
4. Construct Prompts

bash
Copy
Edit
python scripts/construct_prompt.py
5. Run Inference

bash
Copy
Edit
python scripts/ollama_inference.py
6. Evaluate and Visualize Results

bash
Copy
Edit
python scripts/visualise_results.py
Evaluation
Metrics computed:

Accuracy

Precision

Recall

F1-Score

Baseline (zero-shot TinyLLaMA) results are compared with RAEmoLLM-inspired emotion-aware ICL results. Outputs are stored in results/.

Reproducibility Notes
Python 3.10+ recommended

CPU works, but GPU recommended for faster inference

Ollama TinyLLaMA must be pulled locally (ollama pull tinylama)

DistilRoBERTa weights downloaded automatically via transformers

License
This project is licensed under the MIT License. See the LICENSE file for details.

yaml
Copy
Edit

---

If you want, I can **also add a “Scripts Overview” section** explaining what each script does in 2–3 lines so users immediately understand the workflow.  

Do you want me to do that?
