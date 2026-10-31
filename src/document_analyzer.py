
def analyze_documents(sentences_a, sentences_b, matches):
    """
    Generate document-level similarity statistics
    using the sentence matches detected in Phase 6.
    """

    total_sentences_a = len(sentences_a)
    total_sentences_b = len(sentences_b)

    # Handle documents without sentences
    if total_sentences_a == 0 or total_sentences_b == 0:
        raise ValueError(
            "Both documents must contain sentences."
        )

    # Identify unique matched sentence IDs
    matched_ids_a = {
        match["document_a_sentence_id"]
        for match in matches
    }

    matched_ids_b = {
        match["document_b_sentence_id"]
        for match in matches
    }

    # Count unique matched sentences
    matched_sentences_a = len(matched_ids_a)
    matched_sentences_b = len(matched_ids_b)

    # Calculate unmatched sentences
    unmatched_sentences_a = (
        total_sentences_a - matched_sentences_a
    )

    unmatched_sentences_b = (
        total_sentences_b - matched_sentences_b
    )

    # Calculate directional match coverage
    coverage_a = (
        matched_sentences_a / total_sentences_a
    ) * 100

    coverage_b = (
        matched_sentences_b / total_sentences_b
    ) * 100

    # Average of the two directional coverage values.
    # This is a coverage summary, not a plagiarism probability.
    matched_coverage_score = (
        coverage_a + coverage_b
    ) / 2

    # Calculate similarity statistics
    if matches:
        similarity_scores = [
            match["similarity_score"]
            for match in matches
        ]

        average_similarity = (
            sum(similarity_scores) / len(similarity_scores)
        )

        highest_similarity = max(similarity_scores)

    else:
        average_similarity = 0.0
        highest_similarity = 0.0

    # Count match categories
    category_counts = {
        "Very High": 0,
        "High": 0,
        "Moderate": 0,
        "Low": 0
    }

    for match in matches:
        category = match["match_category"]

        if category in category_counts:
            category_counts[category] += 1

    # Final analysis report
    report = {
        "total_sentences_a": total_sentences_a,
        "total_sentences_b": total_sentences_b,

        "matched_sentences_a": matched_sentences_a,
        "matched_sentences_b": matched_sentences_b,

        "unmatched_sentences_a": unmatched_sentences_a,
        "unmatched_sentences_b": unmatched_sentences_b,

        "coverage_a": round(coverage_a, 2),
        "coverage_b": round(coverage_b, 2),

        "matched_coverage_score": round(
            matched_coverage_score, 2
        ),

        "total_matching_pairs": len(matches),

        "average_similarity": round(
            average_similarity * 100, 2
        ),

        "highest_similarity": round(
            highest_similarity * 100, 2
        ),

        "category_counts": category_counts
    }

    return report
