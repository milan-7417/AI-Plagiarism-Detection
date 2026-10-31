---
title: Semantrix
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# 🔍 Semantrix
### AI-Powered Document Similarity & Paraphrase Detection

Semantrix is an AI-powered document analysis application designed to identify textual similarities, paraphrased content, and semantically related sentences across two documents.

It uses **BERT (Bidirectional Encoder Representations from Transformers)** to generate contextual sentence embeddings and cosine similarity to measure semantic similarity between sentences.

The application provides an interactive interface for uploading documents, analyzing matching sentences, and reviewing similarity results.

---

## 🚀 Features

- **Multi-format Document Support:** Upload PDF, DOCX, and TXT documents.
- **BERT-Based Embeddings:** Generate contextual sentence representations using pretrained BERT Base.
- **Semantic Similarity Detection:** Compare sentences using cosine similarity.
- **Paraphrase Detection:** Identify sentences that express similar meanings using different wording.
- **Sentence-Level Matching:** Find matching sentences between two documents.
- **Similarity Threshold:** Customize the similarity threshold for identifying matches.
- **Match Classification:** Categorize detected matches into different similarity categories.
- **Highlighted Results:** Visually highlight matching content for easier analysis.
- **Document Analysis:** View matching statistics and document-level similarity information.
- **Performance Summary:** Review analysis results in an interactive dashboard.
- **CSV Export:** Download detected matches for further analysis.

---

## 🧠 How It Works

Semantrix follows a semantic similarity analysis pipeline.

1. **Document Upload:** Upload two documents for comparison.
2. **Text Extraction:** Extract textual content from PDF, DOCX, or TXT files.
3. **Text Preprocessing:** Split extracted content into individual sentences.
4. **Embedding Generation:** Generate contextual sentence embeddings using BERT Base.
5. **Similarity Calculation:** Calculate cosine similarity between sentence embeddings.
6. **Match Detection:** Identify sentence pairs that meet the selected similarity threshold.
7. **Match Classification:** Classify detected matches based on textual and semantic similarity.
8. **Result Visualization:** Display matching sentences, similarity scores, and analysis summaries.

### Architecture

```text
        Document 1             Document 2
             |                      |
             v                      v
       Text Extraction        Text Extraction
             |                      |
             v                      v
       Preprocessing          Preprocessing
             |                      |
             v                      v
       Sentence Splitting     Sentence Splitting
             |                      |
             v                      v
       BERT Embeddings        BERT Embeddings
             |                      |
             +----------+-----------+
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