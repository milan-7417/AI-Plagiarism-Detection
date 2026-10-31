
import re
import html
from difflib import SequenceMatcher


def tokenize_sentence(text):
    """
    Split text into words, punctuation, and whitespace.
    Keeping whitespace allows us to preserve the original formatting.
    """
    return re.findall(r"\s+|[^\w\s]+|\w+", text, flags=re.UNICODE)


def highlight_sentence_pair(text_a, text_b):
    """
    Compare two sentences and generate HTML with highlights.

    Green: Matching tokens
    Red: Different or unmatched tokens

    Returns:
        highlighted_a, highlighted_b
    """

    tokens_a = tokenize_sentence(text_a)
    tokens_b = tokenize_sentence(text_b)

    # Compare tokens without considering letter case
    normalized_a = [token.casefold() for token in tokens_a]
    normalized_b = [token.casefold() for token in tokens_b]

    matcher = SequenceMatcher(
        None,
        normalized_a,
        normalized_b,
        autojunk=False
    )

    highlighted_a = []
    highlighted_b = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():

        if tag == "equal":

            for token in tokens_a[i1:i2]:
                highlighted_a.append(
                    html.escape(token)
                    if token.isspace()
                    else f'<span class="match-equal">{html.escape(token)}</span>'
                )

            for token in tokens_b[j1:j2]:
                highlighted_b.append(
                    html.escape(token)
                    if token.isspace()
                    else f'<span class="match-equal">{html.escape(token)}</span>'
                )

        elif tag == "replace":

            for token in tokens_a[i1:i2]:
                highlighted_a.append(
                    html.escape(token)
                    if token.isspace()
                    else f'<span class="match-different">{html.escape(token)}</span>'
                )

            for token in tokens_b[j1:j2]:
                highlighted_b.append(
                    html.escape(token)
                    if token.isspace()
                    else f'<span class="match-different">{html.escape(token)}</span>'
                )

        elif tag == "delete":

            for token in tokens_a[i1:i2]:
                highlighted_a.append(
                    html.escape(token)
                    if token.isspace()
                    else f'<span class="match-different">{html.escape(token)}</span>'
                )

        elif tag == "insert":

            for token in tokens_b[j1:j2]:
                highlighted_b.append(
                    html.escape(token)
                    if token.isspace()
                    else f'<span class="match-different">{html.escape(token)}</span>'
                )

    return "".join(highlighted_a), "".join(highlighted_b)


def get_classification_color(classification):
    """
    Return a display color for each classification.
    """

    colors = {
        "Exact Match": "#15803D",
        "Near-Exact Match": "#16A34A",
        "Potential Paraphrase": "#D97706",
        "Semantically Similar": "#2563EB",
        "Unclassified": "#64748B"
    }

    return colors.get(classification, "#64748B")
