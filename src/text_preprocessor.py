
import re
import nltk

from nltk.tokenize import sent_tokenize


# Download sentence tokenizer resources
nltk.download("punkt")
nltk.download("punkt_tab")


def normalize_text(text):
    """
    Normalize extracted document text.
    Preserve punctuation and original word order.
    """

    # Replace multiple whitespace characters with one space
    text = re.sub(r"\s+", " ", text)

    # Remove leading and trailing whitespace
    text = text.strip()

    return text


def split_into_sentences(text):
    """
    Split document text into individual sentences.
    """

    sentences = sent_tokenize(text)

    # Remove empty sentences
    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return sentences


def preprocess_document(text):
    """
    Complete preprocessing pipeline.
    """

    # Step 1: Normalize text
    cleaned_text = normalize_text(text)

    # Step 2: Split into sentences
    sentences = split_into_sentences(cleaned_text)

    # Step 3: Preserve sentence indexes
    processed_sentences = []

    for index, sentence in enumerate(sentences):

        processed_sentences.append({
            "sentence_id": index,
            "text": sentence
        })

    return processed_sentences
