"""
Semantrix BERT embeddings — Hugging Face ZeroGPU compatible.

Import spaces before torch. The model is loaded at module scope and
placed on "cuda"; ZeroGPU intercepts this placement during startup.
Call generate_embeddings only from the @spaces.GPU-decorated Gradio
event handler in app.py.
"""
import spaces  # Must precede torch.
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "google-bert/bert-base-uncased"
BATCH_SIZE = 8
MAX_LENGTH = 128

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading model: {MODEL_NAME}")
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# On ZeroGPU this is intercepted by the spaces package at startup.
# On a local CPU machine it correctly resolves to "cpu".
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"BERT ready on configured device: {device}")


def load_embedding_model():
    """Backward-compatible accessor; model is already loaded at startup."""
    return tokenizer, model, device


def _get_sentences(processed_sentences):
    sentences = []
    for item in processed_sentences or []:
        text = item.get("text", "") if isinstance(item, dict) else str(item)
        text = text.strip()
        if text:
            sentences.append(text)
    return sentences


def generate_embeddings(
    processed_sentences,
    tokenizer_instance=None,
    model_instance=None,
    target_device=None,
):
    """
    Mean-pool non-padding BERT token embeddings and L2-normalize.
    Returns a CPU tensor (N, hidden_size), never a CUDA tensor.
    Must be called inside the ZeroGPU-decorated event function.
    """
    tokenizer_instance = tokenizer_instance or tokenizer
    model_instance = model_instance or model
    target_device = target_device or device

    sentences = _get_sentences(processed_sentences)
    hidden_size = model_instance.config.hidden_size

    if not sentences:
        return torch.empty((0, hidden_size), dtype=torch.float32)

    embeddings = []
    model_instance.eval()

    with torch.inference_mode():
        for start in range(0, len(sentences), BATCH_SIZE):
            batch = sentences[start:start + BATCH_SIZE]
            inputs = tokenizer_instance(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            inputs = {key: value.to(target_device) for key, value in inputs.items()}

            outputs = model_instance(**inputs)
            token_vectors = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(token_vectors.dtype)

            pooled = (token_vectors * mask).sum(dim=1)
            denominator = mask.sum(dim=1).clamp(min=1e-9)
            pooled = pooled / denominator
            pooled = F.normalize(pooled, p=2, dim=1)

            # ZeroGPU decorated functions cross a process boundary:
            # return CPU tensors, not CUDA tensors.
            embeddings.append(pooled.float().cpu())

            del inputs, outputs, token_vectors, mask, pooled, denominator

    return torch.cat(embeddings, dim=0)
