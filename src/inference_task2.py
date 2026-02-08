"""
Task 2: Author Clustering
Given unlabeled text chunks, group them by author identity using clustering
"""
import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter

from src.tokenizer import tokenize
from src.vocabulary import tokens_to_indices
from src.word2vec import SkipGramSampling, get_word_embeddings


def compute_document_embedding(text, word2idx, embeddings):
    """Convert text to embedding vector using TF-IDF weighted average"""
    tokens = tokenize(text)
    if not tokens:
        return np.zeros(embeddings.shape[1])

    indices = tokens_to_indices(tokens, word2idx)

    # Count token frequencies
    token_counts = {}
    for idx in indices:
        token_counts[idx] = token_counts.get(idx, 0) + 1

    total_tokens = len(indices)
    doc_embedding = np.zeros(embeddings.shape[1])

    # Weighted average with TF-IDF
    for idx, count in token_counts.items():
        tf = count / total_tokens
        idf = 1.0 / (1.0 + np.log(1.0 + count))
        weight = tf * idf
        doc_embedding += weight * embeddings[idx].cpu().numpy()

    # Normalize to unit vector
    norm = np.linalg.norm(doc_embedding)
    if norm > 0:
        doc_embedding = doc_embedding / norm

    return doc_embedding


def compute_stylometric_features(text):
    """Extract stylometric features that capture writing style"""
    tokens = tokenize(text)
    if not tokens:
        return np.zeros(10)

    sentences = [s.strip() for s in text.split('.') if s.strip()]

    features = [
        len(tokens) / max(len(sentences), 1) / 50.0,  # avg sentence length
        len(set(tokens)) / len(tokens),  # type-token ratio (vocab richness)
        sum(1 for c in text if c in '.,!?;:') / len(text) * 10,  # punctuation freq
        sum(len(t) for t in tokens) / len(tokens) / 10.0,  # avg word length
        text.count(',') / len(tokens) * 10,  # comma frequency
        text.count('?') / len(tokens) * 20,  # question marks
        text.count('!') / len(tokens) * 20,  # exclamations
        (text.count(':') + text.count(';')) / len(tokens) * 20,  # colons/semicolons
        (text.count('(') + text.count(')')) / len(tokens) * 20,  # parentheses
        (text.count('"') + text.count("'")) / len(tokens) * 10,  # quotes
    ]

    return np.array(features)


def compute_combined_features(text, word2idx, embeddings):
    """Combine word embeddings with stylometric features"""
    # Get semantic representation from word embeddings
    doc_emb = compute_document_embedding(text, word2idx, embeddings)

    # Get stylistic features
    style_features = compute_stylometric_features(text)

    # Concatenate (weighted combination)
    combined = np.concatenate([doc_emb, style_features * 0.3])

    # Normalize to unit vector
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm

    return combined


class KMeans:
    """K-Means clustering implementation"""

    def __init__(self, n_clusters, max_iters=100, n_init=20, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.n_init = n_init
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, X):
        """Fit K-Means to data with multiple random initializations"""
        np.random.seed(self.random_state)

        best_inertia = float('inf')
        best_centroids = None
        best_labels = None

        print(f"    Running {self.n_init} initializations...")

        # Run multiple times with different initializations
        for init_num in range(self.n_init):
            if (init_num + 1) % 5 == 0:
                print(f"      Initialization {init_num + 1}/{self.n_init}")

            centroids, labels, inertia = self._fit_once(X)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels

        print(f"    Best inertia: {best_inertia:.4f}")

        self.centroids = best_centroids
        self.labels = best_labels
        return self

    def _fit_once(self, X):
        """Single K-Means run"""
        n_samples = X.shape[0]

        # K-Means++ initialization (better than random)
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        centroids = X[indices].copy()

        for iteration in range(self.max_iters):
            # Assign points to nearest centroid
            distances = np.zeros((n_samples, self.n_clusters))
            for k in range(self.n_clusters):
                distances[:, k] = np.linalg.norm(X - centroids[k], axis=1)

            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if np.sum(mask) > 0:
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    # Re-initialize empty cluster
                    new_centroids[k] = X[np.random.randint(n_samples)]

            # Check convergence
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break

            centroids = new_centroids

        # Calculate inertia (sum of squared distances to centroids)
        inertia = 0
        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                inertia += np.sum(np.linalg.norm(X[mask] - centroids[k], axis=1) ** 2)

        return centroids, labels, inertia


def cluster_documents(chunks, word2idx, embeddings, num_authors):
    """
    Cluster text chunks by author using K-Means
    Returns cluster assignments as list
    """
    print(f"  Computing features for {len(chunks)} chunks...")

    # Compute features for all chunks
    features = []
    for i, chunk in enumerate(chunks):
        if (i + 1) % 50 == 0 or (i + 1) == len(chunks):
            print(f"    Processed {i + 1}/{len(chunks)} chunks")

        feat = compute_combined_features(chunk, word2idx, embeddings)
        features.append(feat)

    X = np.array(features)
    print(f"  Feature matrix shape: {X.shape}")

    # Run K-Means clustering
    print(f"  Running K-Means with k={num_authors}...")
    kmeans = KMeans(n_clusters=num_authors, max_iters=100, n_init=20)
    kmeans.fit(X)

    return kmeans.labels.tolist()


def main():
    if len(sys.argv) < 3:
        print("Usage: python inference_task2.py <test_file> <output_dir>")
        sys.exit(1)

    test_file = sys.argv[1]
    output_dir = sys.argv[2]

    print("=" * 60)
    print("Task 2: Author Clustering")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    try:
        checkpoint = torch.load('word2vec_model.pt', map_location='cpu')
        word2idx = checkpoint['word2idx']
        idx2word = checkpoint['idx2word']
        embedding_dim = checkpoint['embedding_dim']
        vocab_size = checkpoint['vocab_size']
    except FileNotFoundError:
        print("ERROR: Model file 'word2vec_model.pt' not found!")
        print("Please train the model first using: python main.py data/train_data/")
        sys.exit(1)
    except KeyError as e:
        print(f"ERROR: Model file is missing key: {e}")
        print("Please retrain the model with the updated main.py")
        sys.exit(1)

    model = SkipGramSampling(vocab_size=vocab_size, embedding_dim=embedding_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    embeddings = get_word_embeddings(model)
    print(f"Loaded embeddings: {embeddings.shape}")

    # Load test data
    print(f"\nLoading test data from {test_file}...")
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Test file '{test_file}' not found!")
        sys.exit(1)

    num_authors = test_data['num_authors']
    min_chunks = test_data['min_chunks_per_author']
    chunks = test_data['chunks']

    print(f"Number of authors: {num_authors}")
    print(f"Minimum chunks per author: {min_chunks}")
    print(f"Total chunks: {len(chunks)}")

    # Perform clustering
    print("\nClustering chunks...")
    cluster_labels = cluster_documents(chunks, word2idx, embeddings, num_authors)

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / "task2_predictions.json"

    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cluster_labels, f)

    # Print cluster distribution
    cluster_counts = Counter(cluster_labels)
    print("\nCluster distribution:")
    for cluster_id in sorted(cluster_counts.keys()):
        count = cluster_counts[cluster_id]
        print(f"  Cluster {cluster_id}: {count} chunks")

    print("\n" + "=" * 60)
    print("Task 2 complete!")
    print(f"Results saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()