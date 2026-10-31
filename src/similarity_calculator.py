
import torch
import torch.nn.functional as F


def calculate_similarity(embeddings_a, embeddings_b):
    """
    Calculate cosine similarity between every sentence
    in Document A and every sentence in Document B.
    """

    if embeddings_a.shape[0] == 0 or embeddings_b.shape[0] == 0:
        raise ValueError("Both documents must contain embeddings.")

    embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
    embeddings_b = F.normalize(embeddings_b, p=2, dim=1)

    similarity_matrix = torch.mm(
        embeddings_a,
        embeddings_b.T
    )

    return similarity_matrix
