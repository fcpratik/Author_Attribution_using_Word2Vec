from collections import Counter
import numpy as np


def build_vocab(token_lists, min_freq=5):
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    vocab = [w for w, c in counter.most_common() if c >= min_freq]
    vocab = ["<UNK>"] + vocab

    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    return word2idx, idx2word


def tokens_to_indices(tokens, word2idx):
    return [word2idx.get(t, word2idx["<UNK>"]) for t in tokens]


def subsample_tokens(token_indices, counter, word2idx, threshold=1e-4):
    """
    Subsample frequent words (Mikolov et al., 2013).
    Dramatically reduces pairs from common words like 'the', 'a', 'of'.
    """
    total = sum(counter.values())
    freqs = {}
    for word, idx in word2idx.items():
        f = counter.get(word, 0) / total
        # Probability of KEEPING the word
        freqs[idx] = min(1.0, (np.sqrt(f / threshold) + 1) * (threshold / f)) if f > 0 else 1.0

    rng = np.random.default_rng()
    rand_vals = rng.random(len(token_indices))
    return [t for t, r in zip(token_indices, rand_vals) if r < freqs.get(t, 1.0)]


def generate_skipgram_pairs_array(token_indices, window_size=2):
    """
    Return pairs as a numpy int32 array of shape (N, 2) instead of a list of tuples.
    Much faster to iterate and convert to tensors.
    """
    n = len(token_indices)
    tokens = np.array(token_indices, dtype=np.int32)

    centers = []
    contexts = []

    for i in range(n):
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)
        for j in range(start, end):
            if i != j:
                centers.append(tokens[i])
                contexts.append(tokens[j])

    return np.column_stack([centers, contexts])  # shape (N, 2)


def build_negative_sampling_table(counter, word2idx, table_size=10_000_000):
    """
    Build a unigram table for O(1) negative sampling via random indexing.
    Much faster than np.random.choice with probabilities each time.
    """
    vocab_size = len(word2idx)
    power = np.zeros(vocab_size, dtype=np.float64)

    for word, idx in word2idx.items():
        power[idx] = counter.get(word, 0) ** 0.75

    power /= power.sum()

    # Build a flat table: sample by picking a random index
    table = np.zeros(table_size, dtype=np.int32)
    idx = 0
    cumulative = 0.0
    for i in range(vocab_size):
        cumulative += power[i]
        target = int(cumulative * table_size)
        while idx < target and idx < table_size:
            table[idx] = i
            idx += 1
    # Fill any remaining
    table[idx:] = vocab_size - 1

    return table


def sample_negatives_batch(neg_table, batch_size, num_neg):
    """
    Sample negatives for an entire batch at once using the prebuilt table.
    Returns shape (batch_size, num_neg) — extremely fast.
    """
    indices = np.random.randint(0, len(neg_table), size=(batch_size, num_neg))
    return neg_table[indices]