
import re
from difflib import SequenceMatcher


def normalize_sentence(text):
    """
    Normalize text for lexical comparison.
    Lowercase, remove punctuation, and normalize whitespace.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def calculate_lexical_similarity(text_a, text_b):
    """
    Calculate lexical similarity using SequenceMatcher.
    Returns a score between 0 and 1.
    """

    normalized_a = normalize_sentence(text_a)
    normalized_b = normalize_sentence(text_b)

    if not normalized_a or not normalized_b:
        return 0.0

    score = SequenceMatcher(
        None,
        normalized_a,
        normalized_b
    ).ratio()

    return score


def classify_sentence_pair(
    text_a,
    text_b,
    semantic_score,
    paraphrase_threshold=0.75
):
    """
    Classify a sentence pair using semantic and lexical similarity.

    Categories:
    1. Exact Match
    2. Near-Exact Match
    3. Potential Paraphrase
    4. Semantically Similar
    5. Unclassified
    """

    normalized_a = normalize_sentence(text_a)
    normalized_b = normalize_sentence(text_b)

    lexical_score = calculate_lexical_similarity(
        text_a,
        text_b
    )

    # Exact textual match after normalization
    if normalized_a == normalized_b:
        classification = "Exact Match"

    # Very similar wording
    elif lexical_score >= 0.85:
        classification = "Near-Exact Match"

    # High semantic similarity with different wording
    elif semantic_score >= paraphrase_threshold:
        classification = "Potential Paraphrase"

    # Similar meaning or wording, requiring review
    elif semantic_score >= 0.60 or lexical_score >= 0.50:
        classification = "Semantically Similar"

    else:
        classification = "Unclassified"

    return {
        "classification": classification,
        "lexical_similarity": round(lexical_score, 4),
        "lexical_similarity_percentage": round(
            lexical_score * 100, 2
        )
    }


def classify_matches(matches, paraphrase_threshold=0.75):
    """
    Add classification information to all detected sentence matches.

    Does not modify the original match dictionaries.
    """

    classified_matches = []

    for match in matches:

        result = classify_sentence_pair(
            text_a=match["document_a_sentence"],
            text_b=match["document_b_sentence"],
            semantic_score=match["similarity_score"],
            paraphrase_threshold=paraphrase_threshold
        )

        updated_match = {
            **match,
            **result
        }

        classified_matches.append(updated_match)

    return classified_matches


def generate_classification_summary(matches):
    """
    Count sentence pairs by classification category.
    """

    categories = {
        "Exact Match": 0,
        "Near-Exact Match": 0,
        "Potential Paraphrase": 0,
        "Semantically Similar": 0,
        "Unclassified": 0
    }

    for match in matches:

        classification = match.get("classification")

        if classification in categories:
            categories[classification] += 1

    return categories
