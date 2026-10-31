
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "google-bert/bert-base-uncased"

BATCH_SIZE = 2
MAX_LENGTH = 128


# --------------------------------------------------
# Load pretrained BERT model
# --------------------------------------------------

def load_embedding_model():
    """
    Load the pretrained BERT Base tokenizer and model.

    Uses CUDA if available; otherwise, uses CPU.

    Returns:
        tokenizer
        model
        device
    """

    print("Loading BERT tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading pretrained BERT model...")

    # Load model weights
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True
    )

    # Set evaluation mode
    model.eval()

    # Select available device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Move model to selected device
    model.to(device)

    print(f"BERT model loaded successfully on: {device}")

    return tokenizer, model, device


# --------------------------------------------------
# Generate sentence embeddings
# --------------------------------------------------

def generate_embeddings(
    processed_sentences,
    tokenizer,
    model,
    device
):
    """
    Generate sentence embeddings using pretrained BERT.

    Mean pooling is applied over token embeddings,
    excluding padding tokens.

    Args:
        processed_sentences: List of dictionaries containing
                             sentence text, or list of strings.
        tokenizer: BERT tokenizer.
        model: Pretrained BERT model.
        device: CPU or CUDA device.

    Returns:
        Tensor of normalized sentence embeddings.
        Shape: (number_of_sentences, hidden_size)
    """

    # Handle empty input
    if not processed_sentences:
        hidden_size = model.config.hidden_size

        return torch.empty(
            (0, hidden_size),
            dtype=torch.float32
        )

    # Extract sentence text
    sentences = []

    for item in processed_sentences:

        if isinstance(item, dict):
            sentence = item["text"]
        else:
            sentence = str(item)

        sentence = sentence.strip()

        if sentence:
            sentences.append(sentence)

    # Handle input containing only empty sentences
    if not sentences:
        hidden_size = model.config.hidden_size

        return torch.empty(
            (0, hidden_size),
            dtype=torch.float32
        )

    embeddings_list = []

    model.eval()

    # Disable gradient calculation for inference
    with torch.inference_mode():

        # Process sentences in small batches
        for start in range(0, len(sentences), BATCH_SIZE):

            batch_sentences = sentences[
                start:start + BATCH_SIZE
            ]

            # Tokenize sentences
            inputs = tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            # Move input tensors to selected device
            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            # Forward pass through BERT
            outputs = model(**inputs)

            # Extract token embeddings
            token_embeddings = outputs.last_hidden_state

            # Attention mask excludes padding tokens
            attention_mask = inputs["attention_mask"]

            mask = attention_mask.unsqueeze(-1).to(
                dtype=token_embeddings.dtype
            )

            # Mean pooling
            sum_embeddings = torch.sum(
                token_embeddings * mask,
                dim=1
            )

            sum_mask = torch.clamp(
                mask.sum(dim=1),
                min=1e-9
            )

            sentence_embeddings = (
                sum_embeddings / sum_mask
            )

            # L2 normalization
            sentence_embeddings = F.normalize(
                sentence_embeddings,
                p=2,
                dim=1
            )

            # Move embeddings to CPU to reduce GPU memory usage
            embeddings_list.append(
                sentence_embeddings.cpu()
            )

            # Release temporary tensors
            del inputs
            del outputs
            del token_embeddings
            del sentence_embeddings

    # Combine all batches
    final_embeddings = torch.cat(
        embeddings_list,
        dim=0
    )

    return final_embeddings
