
import os
import time
import html
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from src.document_processor import extract_text
from src.text_preprocessor import preprocess_document

from src.embedding_generator import (
    load_embedding_model,
    generate_embeddings
)

from src.similarity_calculator import calculate_similarity
from src.match_detector import identify_matching_sentences
from src.document_analyzer import analyze_documents

from src.match_classifier import (
    classify_matches,
    generate_classification_summary
)

from src.result_visualizer import (
    highlight_sentence_pair,
    get_classification_color
)


# =========================================================
# 1. APPLICATION CONFIGURATION
# =========================================================

APP_TITLE = "Semantrix"

MODEL_NAME = "google-bert/bert-base-uncased"


# =========================================================
# 2. ADAPTER FOR GRADIO UPLOADED FILES
# =========================================================

class UploadedDocument:
    """
    Adapter that makes a Gradio file path compatible
    with the existing extract_text() function.
    """

    def __init__(self, filepath):

        self.name = Path(filepath).name

        self._content = Path(filepath).read_bytes()

    def getvalue(self):

        return self._content


# =========================================================
# 3. LOAD PRETRAINED BERT
# =========================================================

_model_cache = None


def get_embedding_model():
    """
    Load BERT only once per application process.
    """

    global _model_cache

    if _model_cache is None:

        _model_cache = load_embedding_model()

    return _model_cache


# =========================================================
# 4. HTML MATCH DISPLAY
# =========================================================

def create_match_html(matches):
    """
    Generate visual sentence comparison cards.
    """

    if not matches:

        return """
        <div class="empty-result">
            No matching sentence pairs were found
            at the selected threshold.
        </div>
        """

    cards = []

    for index, match in enumerate(matches, start=1):

        classification = match.get(
            "classification",
            "Unclassified"
        )

        color = get_classification_color(classification)

        semantic_score = match["similarity_percentage"]

        lexical_score = match.get(
            "lexical_similarity_percentage",
            0
        )

        highlighted_a, highlighted_b = (
            highlight_sentence_pair(
                match["document_a_sentence"],
                match["document_b_sentence"]
            )
        )

        classification_html = html.escape(classification)

        card = f"""
        <div class="match-card">

            <div class="match-header">

                <h3>Match {index}</h3>

                <span class="classification-badge"
                      style="background-color:{color}">
                    {classification_html}
                </span>

            </div>

            <div class="metrics-row">

                <div class="metric-box">
                    <span class="metric-label">
                        Semantic Similarity
                    </span>

                    <strong>{semantic_score}%</strong>
                </div>

                <div class="metric-box">
                    <span class="metric-label">
                        Lexical Similarity
                    </span>

                    <strong>{lexical_score}%</strong>
                </div>

            </div>

            <div class="sentence-grid">

                <div class="sentence-box">

                    <h4>Document A</h4>

                    <p class="sentence-id">
                        Sentence ID:
                        {match["document_a_sentence_id"]}
                    </p>

                    <div class="sentence-text">
                        {highlighted_a}
                    </div>

                </div>

                <div class="sentence-box">

                    <h4>Document B</h4>

                    <p class="sentence-id">
                        Sentence ID:
                        {match["document_b_sentence_id"]}
                    </p>

                    <div class="sentence-text">
                        {highlighted_b}
                    </div>

                </div>

            </div>

            <div class="similarity-track">
                <div class="similarity-fill"
                     style="width:{max(0, min(100, semantic_score))}%">
                </div>
            </div>

        </div>
        """

        cards.append(card)

    return "\n".join(cards)


# =========================================================
# 5. MAIN DOCUMENT PROCESSING FUNCTION
# =========================================================

def compare_documents(
    file_a,
    file_b,
    threshold,
    progress=gr.Progress()
):
    """
    Complete Semantrix document comparison pipeline.
    """

    if file_a is None or file_b is None:

        raise gr.Error(
            "Please upload both documents."
        )

    timings = {}

    total_start = time.perf_counter()

    try:

        # -------------------------------------------------
        # STEP 1: DOCUMENT EXTRACTION
        # -------------------------------------------------

        progress(0.05, desc="Extracting document text...")

        start = time.perf_counter()

        uploaded_a = UploadedDocument(file_a)
        uploaded_b = UploadedDocument(file_b)

        document_a = extract_text(uploaded_a)
        document_b = extract_text(uploaded_b)

        timings["Extraction"] = (
            time.perf_counter() - start
        )

        # -------------------------------------------------
        # STEP 2: TEXT PREPROCESSING
        # -------------------------------------------------

        progress(0.15, desc="Preprocessing documents...")

        start = time.perf_counter()

        sentences_a = preprocess_document(document_a)
        sentences_b = preprocess_document(document_b)

        timings["Preprocessing"] = (
            time.perf_counter() - start
        )

        if not sentences_a or not sentences_b:

            raise gr.Error(
                "No readable sentences were found "
                "in one or both documents."
            )

        # -------------------------------------------------
        # STEP 3: LOAD BERT
        # -------------------------------------------------

        progress(0.25, desc="Loading pretrained BERT...")

        start = time.perf_counter()

        tokenizer, model, device = get_embedding_model()

        timings["Model Loading"] = (
            time.perf_counter() - start
        )

        # -------------------------------------------------
        # STEP 4: GENERATE EMBEDDINGS
        # -------------------------------------------------

        progress(0.40, desc="Generating Document A embeddings...")

        start = time.perf_counter()

        embeddings_a = generate_embeddings(
            sentences_a,
            tokenizer,
            model,
            device
        )

        progress(0.55, desc="Generating Document B embeddings...")

        embeddings_b = generate_embeddings(
            sentences_b,
            tokenizer,
            model,
            device
        )

        timings["Embedding Generation"] = (
            time.perf_counter() - start
        )

        # -------------------------------------------------
        # STEP 5: COSINE SIMILARITY
        # -------------------------------------------------

        progress(0.70, desc="Calculating cosine similarity...")

        start = time.perf_counter()

        similarity_matrix = calculate_similarity(
            embeddings_a,
            embeddings_b
        )

        timings["Similarity Calculation"] = (
            time.perf_counter() - start
        )

        # -------------------------------------------------
        # STEP 6: ADVANCED MATCH DETECTION
        # -------------------------------------------------

        progress(0.78, desc="Detecting matching sentences...")

        start = time.perf_counter()

        matches = identify_matching_sentences(
            similarity_matrix,
            sentences_a,
            sentences_b,
            threshold=threshold
        )

        timings["Match Detection"] = (
            time.perf_counter() - start
        )

        # -------------------------------------------------
        # STEP 7: CLASSIFICATION
        # -------------------------------------------------

        progress(0.84, desc="Classifying sentence matches...")

        start = time.perf_counter()

        matches = classify_matches(
            matches,
            paraphrase_threshold=threshold
        )

        classification_summary = (
            generate_classification_summary(matches)
        )

        timings["Classification"] = (
            time.perf_counter() - start
        )

        # -------------------------------------------------
        # STEP 8: DOCUMENT ANALYSIS
        # -------------------------------------------------

        progress(0.91, desc="Analyzing documents...")

        start = time.perf_counter()

        document_analysis = analyze_documents(
            sentences_a,
            sentences_b,
            matches
        )

        timings["Document Analysis"] = (
            time.perf_counter() - start
        )

        timings["Total"] = (
            time.perf_counter() - total_start
        )

        # -------------------------------------------------
        # STEP 9: CREATE SUMMARY
        # -------------------------------------------------

        summary = f"""
# Semantrix — Comparison Report

**Document A:** {html.escape(Path(file_a).name)}

**Document B:** {html.escape(Path(file_b).name)}

**Processing Device:** {html.escape(str(device))}

## Document-Level Statistics

| Metric | Result |
|---|---:|
| Total matching pairs | {document_analysis["total_matching_pairs"]} |
| Document A sentences | {document_analysis["total_sentences_a"]} |
| Document B sentences | {document_analysis["total_sentences_b"]} |
| Matched sentences in A | {document_analysis["matched_sentences_a"]} |
| Matched sentences in B | {document_analysis["matched_sentences_b"]} |
| Document A coverage | {document_analysis["coverage_a"]}% |
| Document B coverage | {document_analysis["coverage_b"]}% |
| Matched coverage summary | {document_analysis["matched_coverage_score"]}% |

## Similarity Statistics

| Metric | Result |
|---|---:|
| Average pair similarity | {document_analysis["average_similarity"]}% |
| Highest pair similarity | {document_analysis["highest_similarity"]}% |
| Selected threshold | {threshold} |

*Coverage is a measure of matched sentence participation, not a probability of plagiarism.*
"""

        # -------------------------------------------------
        # STEP 10: CLASSIFICATION TABLE
        # -------------------------------------------------

        classification_df = pd.DataFrame(
            [
                {
                    "Classification": category,
                    "Number of Pairs": count
                }

                for category, count in (
                    classification_summary.items()
                )
            ]
        )

        # -------------------------------------------------
        # STEP 11: PERFORMANCE TABLE
        # -------------------------------------------------

        performance_df = pd.DataFrame(
            [
                {
                    "Processing Stage": stage,
                    "Time (seconds)": round(duration, 3)
                }

                for stage, duration in timings.items()
            ]
        )

        # -------------------------------------------------
        # STEP 12: MATCH HTML
        # -------------------------------------------------

        matches_html = create_match_html(matches)

        # -------------------------------------------------
        # STEP 13: CSV REPORT
        # -------------------------------------------------

        csv_path = None

        if matches:

            matches_df = pd.DataFrame(matches)

            temp_file = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".csv",
                prefix="semantrix_report_",
                encoding="utf-8-sig",
                newline="",
                delete=False
            )

            csv_path = temp_file.name

            temp_file.close()

            matches_df.to_csv(
                csv_path,
                index=False,
                encoding="utf-8-sig"
            )

        # -------------------------------------------------
        # STEP 14: ADD EXTRACTED TEXT
        # -------------------------------------------------

        progress(1.0, desc="Comparison completed!")

        return (
            summary,
            classification_df,
            matches_html,
            performance_df,
            csv_path,
            document_a,
            document_b
        )

    except gr.Error:

        raise

    except Exception as error:

        raise gr.Error(
            f"Document processing failed: {str(error)}"
        ) from error


# =========================================================
# 6. GRADIO CSS
# =========================================================

CUSTOM_CSS = """
.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
}

#main-title {
    text-align: center;
    padding: 25px 10px 5px;
}

#main-title h1 {
    font-size: 42px;
    font-weight: 800;
    color: #2563EB;
}

#main-title p {
    color: #64748B;
    font-size: 16px;
}

.upload-panel {
    border: 1px solid #CBD5E1 !important;
    border-radius: 12px !important;
    padding: 15px !important;
}

.match-card {
    background: var(--background-fill-primary);
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 18px;
}

.match-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
}

.match-header h3 {
    margin: 0;
}

.classification-badge {
    color: white;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
}

.metrics-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 15px 0;
}

.metric-box {
    background: var(--background-fill-secondary);
    padding: 12px;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    gap: 5px;
}

.metric-label {
    color: var(--body-text-color-subdued);
    font-size: 12px;
}

.metric-box strong {
    font-size: 23px;
}

.sentence-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
}

.sentence-box {
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    padding: 12px;
    min-width: 0;
}

.sentence-id {
    color: var(--body-text-color-subdued);
    font-size: 12px;
}

.sentence-text {
    line-height: 2;
    overflow-wrap: anywhere;
}

.match-equal {
    background: #DCFCE7;
    color: #166534;
    padding: 2px 3px;
    border-radius: 3px;
    font-weight: 600;
}

.match-different {
    background: #FEE2E2;
    color: #991B1B;
    padding: 2px 3px;
    border-radius: 3px;
    font-weight: 600;
}

.similarity-track {
    height: 8px;
    border-radius: 10px;
    background: var(--background-fill-secondary);
    margin-top: 18px;
    overflow: hidden;
}

.similarity-fill {
    height: 100%;
    background: #2563EB;
    border-radius: 10px;
}

.empty-result {
    text-align: center;
    padding: 35px;
    color: var(--body-text-color-subdued);
    border: 1px dashed var(--border-color-primary);
    border-radius: 10px;
}

.footer {
    text-align: center;
    color: #94A3B8;
    padding: 20px;
    font-size: 13px;
}

@media(max-width: 700px) {
    .sentence-grid {
        grid-template-columns: 1fr;
    }

    .metrics-row {
        grid-template-columns: 1fr;
    }
}
"""


# =========================================================
# 7. GRADIO INTERFACE
# =========================================================

with gr.Blocks(
    title="Semantrix | AI Plagiarism Detection",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate"
    ),
    css=CUSTOM_CSS
) as demo:

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    gr.HTML("""
    <div id="main-title">
        <h1>🔍 Semantrix</h1>
        <p>
            AI-Powered Plagiarism & Paraphrase Detection System
        </p>
    </div>
    """)

    gr.Markdown(
        """
        Upload two documents to identify potentially similar
        sentences, analyze semantic similarity and review
        lexical differences.
        """
    )

    # -----------------------------------------------------
    # DOCUMENT UPLOAD
    # -----------------------------------------------------

    with gr.Row():

        with gr.Column():

            gr.Markdown("### 📄 Document A")

            file_a = gr.File(
                label="Upload first document",
                file_types=[".pdf", ".docx", ".txt"],
                type="filepath"
            )

        with gr.Column():

            gr.Markdown("### 📄 Document B")

            file_b = gr.File(
                label="Upload second document",
                file_types=[".pdf", ".docx", ".txt"],
                type="filepath"
            )

    # -----------------------------------------------------
    # DETECTION SETTINGS
    # -----------------------------------------------------

    gr.Markdown("### Detection Settings")

    threshold = gr.Slider(
        minimum=0.50,
        maximum=1.00,
        value=0.75,
        step=0.01,
        label="Sentence Similarity Threshold",
        info=(
            "Sentence pairs with cosine similarity "
            "at or above this threshold are reported."
        )
    )

    compare_button = gr.Button(
        "🔍 Compare Documents",
        variant="primary",
        size="lg"
    )

    # -----------------------------------------------------
    # OUTPUT TABS
    # -----------------------------------------------------

    gr.Markdown("---")

    gr.Markdown("## Comparison Results")

    with gr.Tab("Overview"):

        summary_output = gr.Markdown(
            value="Upload two documents and click Compare Documents."
        )

        classification_output = gr.Dataframe(
            headers=[
                "Classification",
                "Number of Pairs"
            ],
            label="Classification Summary",
            interactive=False
        )

    with gr.Tab("Visual Sentence Comparison"):

        gr.Markdown(
            """
            **Green:** Matching tokens

            **Red:** Different or unmatched tokens

            Note: Different wording does not necessarily mean
            different meaning.
            """
        )

        matches_output = gr.HTML(
            value="<div class='empty-result'>No results yet.</div>"
        )

    with gr.Tab("Performance"):

        performance_output = gr.Dataframe(
            label="Processing Performance",
            interactive=False
        )

    with gr.Tab("Extracted Text"):

        with gr.Accordion("Document A", open=False):

            document_a_output = gr.Textbox(
                label="Extracted text from Document A",
                lines=15,
                interactive=False
            )

        with gr.Accordion("Document B", open=False):

            document_b_output = gr.Textbox(
                label="Extracted text from Document B",
                lines=15,
                interactive=False
            )

    # -----------------------------------------------------
    # DOWNLOAD REPORT
    # -----------------------------------------------------

    gr.Markdown("---")

    gr.Markdown("### Download Report")

    csv_output = gr.File(
        label="Matching Results CSV",
        interactive=False
    )

    # -----------------------------------------------------
    # EVENT HANDLER
    # -----------------------------------------------------

    compare_button.click(
        fn=compare_documents,
        inputs=[
            file_a,
            file_b,
            threshold
        ],
        outputs=[
            summary_output,
            classification_output,
            matches_output,
            performance_output,
            csv_output,
            document_a_output,
            document_b_output
        ],
        show_progress="full"
    )

    # -----------------------------------------------------
    # FOOTER
    # -----------------------------------------------------

    gr.HTML("""
    <div class="footer">
        Semantrix | AI-Powered Plagiarism & Paraphrase Detection
        <br>
        Built using Python, PyTorch, BERT and Gradio
    </div>
    """)


# =========================================================
# 8. LAUNCH APPLICATION
# =========================================================

if __name__ == "__main__":

    demo.launch()
