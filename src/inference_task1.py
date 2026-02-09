import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter

from src.tokenizer import tokenize
from src.vocabulary import tokens_to_indices
from src.word2vec import SkipGramSampling


def compute_document_embedding(text, word2idx, embeddings_np, idf_weights):

    tokens = tokenize(text)
    dim = embeddings_np.shape[1]
    if not tokens:
        return np.zeros(dim)

    indices = tokens_to_indices(tokens, word2idx)
    total = len(indices)

    # Count term frequencies in this document
    tf_counts = Counter(indices)

    doc_embedding = np.zeros(dim)
    total_weight = 0.0

    for idx, count in tf_counts.items():
        tf = count / total
        idf = idf_weights.get(idx, 1.0)
        weight = tf * idf
        doc_embedding += weight * embeddings_np[idx]
        total_weight += weight

    if total_weight > 0:
        doc_embedding /= total_weight

    # L2 normalize
    norm = np.linalg.norm(doc_embedding)
    if norm > 0:
        doc_embedding /= norm

    return doc_embedding


def compute_stylometric_features(text):
    """
    Stylometric features are used to  capture  author's writing fingerprint:
    punctuation habits, sentence structure, vocabulary richness,...
    """
    tokens = tokenize(text)
    if not tokens:
        return np.zeros(15)

    n_tokens = len(tokens)
    n_chars = len(text)

    # Sentences (rough split)
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    n_sentences = max(len(sentences), 1)

    features = []

    # Average  length of sentence (in tokens)
    features.append((n_tokens / n_sentences) / 50.0)

    # Vocabulary richness (type-token ratio)
    features.append(len(set(tokens)) / n_tokens)

    #  Average word length
    features.append(sum(len(t) for t in tokens) / n_tokens / 10.0)

    #  Punctuation frequencies (strong author signal!)
    for char in [',', ';', ':', '!', '?', '-']:
        features.append(text.count(char) / n_tokens * 10)

    #  Parentheses frequency
    features.append((text.count('(') + text.count(')')) / n_tokens * 20)

    #  Quote frequency
    features.append((text.count('"') + text.count("'")) / n_tokens * 10)

    #  All-caps word ratio (emphasis style)
    all_caps = sum(1 for t in text.split() if t.isupper() and len(t) > 1)
    features.append(all_caps / n_tokens * 10)

    #  Short word ratio (<=3 chars) — function word density
    short = sum(1 for t in tokens if len(t) <= 3)
    features.append(short / n_tokens)

    #  Long word ratio (>=8 chars) — vocabulary sophistication
    long_ = sum(1 for t in tokens if len(t) >= 8)
    features.append(long_ / n_tokens)

    #  Paragraph count (formatting style)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    features.append(len(paragraphs) / max(n_sentences, 1))

    return np.array(features)


def compute_combined_features(text, word2idx, embeddings_np, idf_weights, style_weight=0.5):
    """Combine TF-IDF embeddings + stylometric features into one single vector."""
    doc_emb = compute_document_embedding(text, word2idx, embeddings_np, idf_weights)
    style_features = compute_stylometric_features(text)

    combined = np.concatenate([doc_emb, style_features * style_weight])

    norm = np.linalg.norm(combined)
    if norm > 0:
        combined /= norm

    return combined


def build_idf_weights(train_dir, word2idx):
    """
    Build IDF weights from training corpus.
    Words appearing in fewer documents get higher weight → better at distinguishing authors.
    """
    txt_files = sorted(Path(train_dir).glob("*.txt"))
    num_docs = len(txt_files)

    doc_freq = Counter()  # How many documents contain each word index

    for filepath in txt_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = tokenize(text)
        indices = set(tokens_to_indices(tokens, word2idx))
        doc_freq.update(indices)

    # IDF = log(N / df)
    idf = {}
    for idx in range(len(word2idx)):
        df = doc_freq.get(idx, 0)
        idf[idx] = np.log(num_docs / (1 + df)) + 1.0  # smoothed IDF

    return idf


def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    n1 = np.linalg.norm(vec1)
    n2 = np.linalg.norm(vec2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def rank_candidates(query_text, candidates, word2idx, embeddings_np, idf_weights, style_weight=0.5):
    """
    Rank candidates by similarity using combined TF-IDF embeddings + stylometric features.
    """
    query_vec = compute_combined_features(query_text, word2idx, embeddings_np, idf_weights, style_weight)

    scores = []
    for cand_id, cand_text in candidates.items():
        cand_vec = compute_combined_features(cand_text, word2idx, embeddings_np, idf_weights, style_weight)
        score = cosine_similarity(query_vec, cand_vec)
        scores.append((cand_id, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m src.inference_task1 <test_file> <output_dir>")
        sys.exit(1)

    test_file = sys.argv[1]
    output_dir = sys.argv[2]

    # ---- Load Model & Vocabulary ----
    print("\nLoading model...")
    checkpoint = torch.load('word2vec_model.pt', map_location='cpu')
    word2idx = checkpoint['word2idx']
    embedding_dim = checkpoint['embedding_dim']
    vocab_size = len(word2idx)

    model = SkipGramSampling(vocab_size=vocab_size, embedding_dim=embedding_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    embeddings_np = model.in_embed.weight.data.numpy()
    print(f"Loaded embeddings: {embeddings_np.shape}")

    # ---- Build IDF weights from training data ----
    train_dir = checkpoint['hyperparameters'].get('train_dir', 'data/train_data')

    train_path = Path(train_dir)
    if not train_path.exists():
        train_path = Path('data/train_data')

    if train_path.exists():
        print(f"Building IDF weights from {train_path}...")
        idf_weights = build_idf_weights(train_path, word2idx)
    else:
        print("Warning: training data not found, using uniform IDF weights")
        idf_weights = {i: 1.0 for i in range(vocab_size)}

    # ---- Load Test Data ----
    print(f"\nLoading test data from {test_file}...")
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    print(f"Found {len(test_data)} test cases.")

    # ---- Process & Write Results ----
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "task1_predictions.jsonl"
    print(f"Processing and saving to {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for i, case in enumerate(test_data):
            query_id = case['query_id']
            query_text = case['query_text']
            candidates = case['candidates']

            ranked = rank_candidates(query_text, candidates, word2idx, embeddings_np, idf_weights)
            ranked_ids = [cand_id for cand_id, score in ranked]

            result = {
                "query_id": query_id,
                "ranked_candidates": ranked_ids
            }
            f_out.write(json.dumps(result) + '\n')

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(test_data)}...")

    print("=" * 60)
    print("Task 1 complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()