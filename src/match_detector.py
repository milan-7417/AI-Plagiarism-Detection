
import torch


def identify_matching_sentences(
    similarity_matrix,
    sentences_a,
    sentences_b,
    threshold=0.75
):
    """
    Identify all sentence pairs whose cosine similarity
    is greater than or equal to the threshold.

    Features:
    1. Multiple sentence matching
    2. Forward matching (A -> B)
    3. Reverse matching (B -> A)
    4. Duplicate match handling
    5. Similarity categorization
    """

    if similarity_matrix.numel() == 0:
        return []

    if similarity_matrix.shape != (
        len(sentences_a),
        len(sentences_b)
    ):
        raise ValueError(
            "Similarity matrix dimensions do not match "
            "the number of sentences."
        )

    # Convert similarity matrix into a list
    similarity_matrix = similarity_matrix.cpu()

    matches = []

    # Track matched sentence pairs to avoid duplicates
    matched_pairs = set()

    # -------------------------------------------------
    # 1. FORWARD MATCHING: DOCUMENT A -> DOCUMENT B
    # -------------------------------------------------

    for index_a in range(len(sentences_a)):

        for index_b in range(len(sentences_b)):

            score = similarity_matrix[index_a, index_b].item()

            if score >= threshold:

                pair = (index_a, index_b)

                if pair not in matched_pairs:

                    matched_pairs.add(pair)

                    matches.append({
                        "document_a_sentence_id": index_a,
                        "document_a_sentence":
                            sentences_a[index_a]["text"],

                        "document_b_sentence_id": index_b,
                        "document_b_sentence":
                            sentences_b[index_b]["text"],

                        "similarity_score": round(score, 4),

                        "similarity_percentage":
                            round(score * 100, 2),

                        "match_category":
                            categorize_similarity(score)
                    })

    # -------------------------------------------------
    # 2. REVERSE MATCHING: DOCUMENT B -> DOCUMENT A
    # -------------------------------------------------

    best_scores_b, best_indices_a = torch.max(
        similarity_matrix,
        dim=0
    )

    for index_b, (score, index_a) in enumerate(
        zip(
            best_scores_b.tolist(),
            best_indices_a.tolist()
        )
    ):

        if score >= threshold:

            pair = (index_a, index_b)

            if pair not in matched_pairs:

                matched_pairs.add(pair)

                matches.append({
                    "document_a_sentence_id": index_a,
                    "document_a_sentence":
                        sentences_a[index_a]["text"],

                    "document_b_sentence_id": index_b,
                    "document_b_sentence":
                        sentences_b[index_b]["text"],

                    "similarity_score": round(score, 4),

                    "similarity_percentage":
                        round(score * 100, 2),

                    "match_category":
                        categorize_similarity(score)
                })

    # -------------------------------------------------
    # 3. SORT MATCHES BY SIMILARITY
    # -------------------------------------------------

    matches.sort(
        key=lambda item: item["similarity_score"],
        reverse=True
    )

    return matches


# -----------------------------------------------------
# 4. SIMILARITY CATEGORIZATION
# -----------------------------------------------------

def categorize_similarity(score):
    """
    Categorize a similarity score.

    These are configurable reporting bands,
    not proof of plagiarism.
    """

    if score >= 0.90:
        return "Very High"

    elif score >= 0.80:
        return "High"

    elif score >= 0.70:
        return "Moderate"

    else:
        return "Low"
