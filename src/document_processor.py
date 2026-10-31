
import pymupdf
from docx import Document
from pathlib import Path
from io import BytesIO


def extract_pdf(file_bytes):
    """Extract text from uploaded PDF bytes."""

    text = []

    with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
        for page in pdf:
            text.append(page.get_text("text"))

    return "\n".join(text)


def extract_docx(file_bytes):
    """Extract text from uploaded DOCX bytes."""

    document = Document(BytesIO(file_bytes))

    paragraphs = [
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    # Extract text from tables
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    paragraphs.append(cell.text)

    return "\n".join(paragraphs)


def extract_txt(file_bytes):
    """Extract text from uploaded TXT bytes."""

    return file_bytes.decode("utf-8-sig", errors="replace")


def extract_text(uploaded_file):
    """
    Extract text from a Streamlit uploaded file.

    Supported formats: PDF, DOCX, TXT.
    """

    filename = getattr(uploaded_file, "name", str(uploaded_file))
    extension = Path(filename).suffix.lower()

    # Read uploaded file content
    if hasattr(uploaded_file, "getvalue"):
        file_bytes = uploaded_file.getvalue()
    else:
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()

    if not file_bytes:
        raise ValueError("The uploaded document is empty.")

    if extension == ".pdf":
        text = extract_pdf(file_bytes)

    elif extension == ".docx":
        text = extract_docx(file_bytes)

    elif extension == ".txt":
        text = extract_txt(file_bytes)

    else:
        raise ValueError(
            f"Unsupported format: {extension}. "
            "Upload PDF, DOCX or TXT."
        )

    if not text.strip():
        raise ValueError(
            "No extractable text found. "
            "The document may be empty or scanned."
        )

    return text
