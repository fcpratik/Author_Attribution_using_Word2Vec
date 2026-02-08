import json
import torch
import numpy as np


def save_vocab(word2idx, filepath):
    with open(filepath, 'w') as f:
        json.dump(word2idx, f)


def load_vocab(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def get_text_embedding(model, word2idx, text, tokenizer_func):
    """
    Computes the average word vector for a given text.
    """
    tokens = tokenizer_func(text)
    indices = [word2idx.get(t, word2idx.get("<UNK>", 0)) for t in tokens]

    if not indices:
        return np.zeros(model.in_embed.embedding_dim)

    # Convert to tensor
    input_tensor = torch.LongTensor(indices)

    # Get embeddings (using the input embedding matrix)
    # Ensure model is on CPU/Evaluation mode
    with torch.no_grad():
        embeddings = model.in_embed(input_tensor)

    # Average the vectors (Mean Pooling)
    mean_embedding = torch.mean(embeddings, dim=0).numpy()

    # L2 Normalize for Cosine Similarity later
    norm = np.linalg.norm(mean_embedding)
    if norm > 0:
        mean_embedding = mean_embedding / norm

    return mean_embedding