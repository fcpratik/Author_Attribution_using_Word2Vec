"""
Task 1: Author Verification and Ranking
Given a query text and candidate texts, rank candidates by similarity to query author's style
"""
import sys
import json
import torch
import numpy as np
from pathlib import Path

from src.tokenizer import tokenize
from src.vocabulary import tokens_to_indices
from src.word2vec import SkipGramSampling, get_word_embeddings

import torch.nn.functional as F

def compute_document_embedding(text, word2idx, embeddings):
    """
    Convert a text document to a single embedding vector
    Uses average of word embeddings with TF-IDF-like weighting
    """
    tokens = tokenize(text)
    if not tokens:
        return np.zeros(embeddings.shape[1])

    indices = tokens_to_indices(tokens, word2idx)

    # Get embeddings for all tokens
    token_embeddings = []
    token_counts = {}

    for idx in indices:
        if idx in token_counts:
            token_counts[idx] += 1
        else:
            token_counts[idx] = 1

    # Weighted average: give more weight to rare words
    total_tokens = len(indices)
    doc_embedding = np.zeros(embeddings.shape[1])

    for idx, count in token_counts.items():
        # TF weight (normalized frequency)
        tf = count / total_tokens
        # Simple IDF approximation: rare words get higher weight
        idf = 1.0 / (1.0 + np.log(1.0 + count))
        weight = tf * idf

        doc_embedding += weight * embeddings[idx].numpy()

    # Normalize
    norm = np.linalg.norm(doc_embedding)
    if norm > 0:
        doc_embedding = doc_embedding / norm

    return doc_embedding


def compute_stylometric_features(text):
    """
    Extract stylometric features that capture author's writing style
    Returns a feature vector
    """
    tokens = tokenize(text)
    if not tokens:
        return np.zeros(10)

    # Split into sentences (rough approximation)
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]

    features = []

    # 1. Average sentence length
    avg_sent_len = len(tokens) / max(len(sentences), 1)
    features.append(avg_sent_len / 50.0)  # Normalize

    # 2. Vocabulary richness (type-token ratio)
    unique_tokens = len(set(tokens))
    ttr = unique_tokens / len(tokens) if tokens else 0
    features.append(ttr)

    # 3. Punctuation frequency
    punct_count = sum(1 for char in text if char in '.,!?;:')
    punct_freq = punct_count / len(text) if text else 0
    features.append(punct_freq * 10)  # Scale up

    # 4. Average word length
    avg_word_len = sum(len(token) for token in tokens) / len(tokens) if tokens else 0
    features.append(avg_word_len / 10.0)  # Normalize

    # 5. Comma frequency
    comma_freq = text.count(',') / len(tokens) if tokens else 0
    features.append(comma_freq * 10)

    # 6. Question mark frequency
    question_freq = text.count('?') / len(tokens) if tokens else 0
    features.append(question_freq * 20)

    # 7. Exclamation frequency
    exclaim_freq = text.count('!') / len(tokens) if tokens else 0
    features.append(exclaim_freq * 20)

    # 8. Colon/semicolon frequency
    colon_freq = (text.count(':') + text.count(';')) / len(tokens) if tokens else 0
    features.append(colon_freq * 20)

    # 9. Parentheses frequency
    paren_freq = (text.count('(') + text.count(')')) / len(tokens) if tokens else 0
    features.append(paren_freq * 20)

    # 10. Quote frequency
    quote_freq = (text.count('"') + text.count("'")) / len(tokens) if tokens else 0
    features.append(quote_freq * 10)

    return np.array(features)


def compute_combined_features(text, word2idx, embeddings):
    """Combine word embeddings with stylometric features"""
    # Get word embedding representation
    doc_emb = compute_document_embedding(text, word2idx, embeddings)

    # Get stylometric features
    style_features = compute_stylometric_features(text)

    # Concatenate (give more weight to embeddings)
    combined = np.concatenate([doc_emb, style_features * 0.3])

    # Normalize
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm

    return combined


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def get_text_vector(text, word2idx, embeddings):
    tokens = tokenize(text)  # Make sure you import tokenize
    if not tokens:
        return torch.zeros(embeddings.size(1))

    indices = [word2idx.get(t, 0) for t in tokens]  # 0 is <UNK>
    indices_tensor = torch.tensor(indices, dtype=torch.long)

    # Select vectors and average them
    # shape: (num_words, dim)
    vectors = embeddings[indices_tensor]

    # Average pooling
    return torch.mean(vectors, dim=0)


def rank_candidates(query_text, candidates, word2idx, embeddings):
    query_vec = get_text_vector(query_text, word2idx, embeddings)

    scores = []
    for cand_id, cand_text in candidates.items():
        cand_vec = get_text_vector(cand_text, word2idx, embeddings)

        # Cosine Similarity
        # dim=0 because these are 1D vectors
        score = F.cosine_similarity(query_vec, cand_vec, dim=0).item()
        scores.append((cand_id, score))

    # Sort descending (highest score first)
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m src.inference_task1 <test_file> <output_dir>")
        sys.exit(1)

    test_file = sys.argv[1]
    output_dir = sys.argv[2]

    # 1. Load Model & Vocabulary
    print("\nLoading model...")
    checkpoint = torch.load('word2vec_model.pt')  # Ensure filename matches what you saved
    word2idx = checkpoint['word2idx']  # Ensure these keys match your save dictionary
    # idx2word = checkpoint['idx2word']    # Not strictly needed for inference
    embedding_dim = checkpoint['embedding_dim']  # Or load from checkpoint if you saved it
    vocab_size = len(word2idx)

    model = SkipGramSampling(vocab_size=vocab_size, embedding_dim=embedding_dim)
    model.load_state_dict(checkpoint['model_state_dict'])  # Adjust key if needed
    model.eval()

    # Get embeddings matrix once
    embeddings = model.in_embed.weight.data
    print(f"Loaded embeddings: {embeddings.shape}")

    # 2. Load Test Data
    print(f"\nLoading test data from {test_file}...")
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)  # This is a LIST of dictionaries

    print(f"Found {len(test_data)} test cases.")

    # 3. Prepare Output File
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "task1_predictions.jsonl"

    # 4. Processing Loop
    print(f"Processing and saving to {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f_out:

        # LOOP through each test case in the list
        for i, case in enumerate(test_data):

            # Extract data for THIS specific case
            query_id = case['query_id']
            query_text = case['query_text']
            candidates = case['candidates']  # Dictionary of candidates

            # --- Perform Ranking (Logic you likely have in a helper function) ---
            # Assuming rank_candidates returns a list like [('cand_1', 0.9), ('cand_2', 0.4)]
            # You need to implement rank_candidates or put the logic here.
            ranked = rank_candidates(query_text, candidates, word2idx, embeddings)

            # Extract just the IDs
            ranked_ids = [cand_id for cand_id, score in ranked]

            # Create result object
            result = {
                "query_id": query_id,
                "ranked_candidates": ranked_ids
            }

            # Write to file immediately (JSONL format = one JSON per line)
            f_out.write(json.dumps(result) + '\n')

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(test_data)}...")

    print("=" * 60)
    print("Task 1 complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
