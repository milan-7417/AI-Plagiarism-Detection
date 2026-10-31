# 🔍 Semantrix
### AI-Powered Document Similarity & Paraphrase Detection



<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?style=flat-square&logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Transformers-Hugging%20Face-yellow?style=flat-square&logo=huggingface" alt="Transformers"/>
  <img src="https://img.shields.io/badge/Gradio-Interface-orange?style=flat-square&logo=gradio" alt="Gradio"/>
  <img src="https://img.shields.io/badge/ZeroGPU-Deployment-green?style=flat-square" alt="ZeroGPU"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"/>
</p>

---

## 📌 Overview

**Semantrix** is an AI-powered document similarity and paraphrase detection application that analyzes two documents to identify matching sentences, paraphrased content, and semantically related text.

Unlike traditional text-matching approaches that primarily depend on exact word matches, Semantrix uses **BERT (Bidirectional Encoder Representations from Transformers)** to generate contextual sentence embeddings and identify similarities based on semantic meaning.

The application provides sentence-level comparison, similarity scores, match classification, highlighted results, and document-level analysis through an interactive Gradio interface.

## 🚀 Live Demo

Try Semantrix directly without installing anything:

**[Launch Semantrix on Hugging Face Spaces →](https://huggingface.co/spaces/milan-123/Semantrix)**

## ✨ Key Features

- 📄 **Multi-format Document Support:** Upload PDF, DOCX, and TXT documents.
- 🧠 **BERT-Based Embeddings:** Generate contextual sentence representations using pretrained BERT Base.
- 🔍 **Semantic Similarity Detection:** Identify sentences with similar meanings, even when their wording differs.
- 🔄 **Paraphrase Detection:** Find potentially paraphrased sentences across documents.
- 🎯 **Sentence-Level Matching:** Identify matching sentence pairs between two documents.
- ⚙️ **Adjustable Similarity Threshold:** Customize the sensitivity of similarity detection.
- 🏷️ **Match Classification:** Categorize detected matches using textual and semantic similarity.
- 🎨 **Highlighted Results:** Visually highlight matching content for easier interpretation.
- 📊 **Document Analysis:** Analyze matching statistics and document-level coverage.
- 📥 **CSV Export:** Download detected matches for further analysis.
- ⚡ **GPU Acceleration:** Supports GPU-based inference through Hugging Face ZeroGPU.

## 🖥️ Application Preview


<p align="center">
  <img src=assets\home (3).png width="85%" alt="Semantrix Home Interface"/>
</p>

<p align="center">
  <em>Semantrix application interface</em>
</p>

> Replace `assets/home.png` with the actual path of your screenshot, or add more screenshots of the results dashboard and document comparison interface.

## 🧠 How Semantrix Works

Semantrix follows a document similarity analysis pipeline:

1. **Document Upload:** Upload two documents for comparison.
2. **Text Extraction:** Extract text from PDF, DOCX, or TXT files.
3. **Text Preprocessing:** Process extracted content and split it into individual sentences.
4. **Embedding Generation:** Generate contextual sentence embeddings using pretrained BERT.
5. **Similarity Calculation:** Calculate cosine similarity between sentence embeddings.
6. **Match Detection:** Identify sentence pairs that meet the selected similarity threshold.
7. **Match Classification:** Classify detected matches based on textual and semantic similarity.
8. **Result Visualization:** Display matching sentences, similarity scores, classifications, and analysis summaries.

## 🏗️ System Architecture

```text
             DOCUMENT 1                  DOCUMENT 2
                 |                           |
                 v                           v
          Text Extraction              Text Extraction
                 |                           |
                 v                           v
          Text Preprocessing           Text Preprocessing
                 |                           |
                 v                           v
          Sentence Splitting           Sentence Splitting
                 |                           |
                 v                           v
          BERT Embeddings               BERT Embeddings
                 |                           |
                 +-------------+-------------+
                               |
                               v
                      Cosine Similarity
                               |
                               v
                        Match Detection
                               |
                               v
                     Match Classification
                               |
                               v
                     Result Visualization
                               |
                               v
                       Analysis Report
```

## 🤖 Pretrained Language Model

Semantrix uses the following pretrained model:

| Property | Details |
|---|---|
| Model | BERT Base Uncased |
| Model Identifier | `google-bert/bert-base-uncased` |
| Architecture | Transformer Encoder |
| Model Type | Bidirectional Language Model |
| Embedding Dimension | 768 |
| Model Library | Hugging Face Transformers |

**Model:** [BERT Base Uncased on Hugging Face](https://huggingface.co/google-bert/bert-base-uncased)

The model generates contextual token representations. Mean pooling over the attention mask is used to obtain sentence-level embeddings, followed by L2 normalization.

## 📐 Similarity Calculation

Semantrix uses cosine similarity to measure the similarity between two sentence embedding vectors.

The cosine similarity formula is:

\[
\operatorname{Similarity}(A,B)=
\frac{A \cdot B}{\|A\|\|B\|}
\]

Where:

- \(A\) represents the embedding of a sentence from Document 1.
- \(B\) represents the embedding of a sentence from Document 2.
- \(A \cdot B\) represents the dot product.
- \(\|A\|\) and \(\|B\|\) represent the vector magnitudes.

A higher cosine similarity score indicates that two sentence embeddings are more closely aligned in the embedding space.

## 🏷️ Match Classification

Semantrix classifies detected sentence pairs using textual and semantic similarity signals.

| Classification | Description |
|---|---|
| Exact Match | Sentences with identical or nearly identical wording |
| Near Exact Match | Sentences with minor textual differences |
| Potential Paraphrase | Sentences expressing similar meanings using different wording |
| Semantic Similarity | Sentences sharing related semantic information |

*Classification is heuristic and should be interpreted alongside the similarity score and original text.*

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend / UI | Gradio |
| Deep Learning Framework | PyTorch |
| NLP Model | BERT Base Uncased |
| Model Library | Hugging Face Transformers |
| Text Processing | NLTK |
| Similarity Calculation | PyTorch, Scikit-learn |
| Data Processing | NumPy, Pandas |
| PDF Processing | PyMuPDF |
| Word Document Processing | python-docx |
| Deployment | Hugging Face Spaces |
| GPU Infrastructure | Hugging Face ZeroGPU |

## 📂 Project Structure

```text
AI-Plagiarism-Detection/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── document_processor.py
│   ├── text_preprocessor.py
│   ├── embedding_generator.py
│   ├── similarity_calculator.py
│   ├── match_detector.py
│   ├── document_analyzer.py
│   ├── match_classifier.py
│   └── result_visualizer.py
│
└── assets/
    └── home.png
```

### Module Description

| Module | Responsibility |
|---|---|
| `app.py` | Gradio interface and application workflow |
| `document_processor.py` | Extract text from uploaded documents |
| `text_preprocessor.py` | Preprocess text and split it into sentences |
| `embedding_generator.py` | Generate contextual BERT embeddings |
| `similarity_calculator.py` | Calculate cosine similarity |
| `match_detector.py` | Identify matching sentence pairs |
| `document_analyzer.py` | Calculate document-level analysis metrics |
| `match_classifier.py` | Classify detected sentence matches |
| `result_visualizer.py` | Generate highlighted comparison results |

## ⚙️ Installation and Setup

Follow these steps to run Semantrix locally.

### 1. Clone the repository

```bash
git clone https://github.com/milan-7417/AI-Plagiarism-Detection.git
```

### 2. Navigate to the project directory

```bash
cd AI-Plagiarism-Detection
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Download NLTK resources

```python
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
```

### 6. Run the application

```bash
python app.py
```

Open the local Gradio URL displayed in your terminal.

## 📦 Dependencies

The project uses the following major libraries:

```text
gradio
spaces
torch
transformers
accelerate
huggingface-hub
pandas
numpy
scikit-learn
pymupdf
python-docx
nltk
```

For the exact dependency versions, refer to [`requirements.txt`](requirements.txt).

## 🌐 Deployment

Semantrix is deployed using **Hugging Face Spaces** with Gradio and ZeroGPU support.

| Platform | Link |
|---|---|
| Live Application | [Semantrix](https://huggingface.co/spaces/milan-123/Semantrix) |
| Source Code | [GitHub Repository](https://github.com/milan-7417/AI-Plagiarism-Detection) |
| Pretrained Model | [BERT Base Uncased](https://huggingface.co/google-bert/bert-base-uncased) |

## ⚠️ Limitations and Disclaimer

- Similarity scores are indicators of textual or semantic similarity, not definitive proof of plagiarism.
- Common phrases and domain-specific terminology may produce high similarity scores.
- Results depend on document extraction quality and sentence segmentation.
- BERT has a maximum input sequence length, so longer sentences may be truncated during tokenization.
- Semantic similarity should be evaluated with document context, citations, and human judgment.

**Semantrix is intended as a document similarity analysis and research-support tool, not as an automated plagiarism verdict system.**

## 🔮 Future Improvements

- Support for additional multilingual transformer models.
- Improved paraphrase detection using fine-tuned sentence embedding models.
- Document-level similarity visualization.
- Advanced reporting and downloadable PDF reports.
- Improved detection of cross-document matching patterns.
- Support for additional document formats.

## 👨‍💻 Author

**Milan Kumar**

B.Tech – Computer Science and Engineering  
Specialization: Artificial Intelligence & Machine Learning

<p>
  <a href="https://github.com/milan-7417">
    <img src="https://img.shields.io/badge/GitHub-milan--7417-black?style=for-the-badge&logo=github" alt="GitHub"/>
  </a>
  <a href="https://huggingface.co/milan-123">
    <img src="https://img.shields.io/badge/Hugging%20Face-milan--123-yellow?style=for-the-badge&logo=huggingface" alt="Hugging Face"/>
  </a>
</p>

---

<p align="center">
  <strong>🔍 Semantrix — Understand Similarity Beyond Words.</strong>
</p>