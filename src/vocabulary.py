from collections import Counter
import numpy as np

def build_vocab(token_lists, min_freq=5):
    counter = Counter()

    for tokens in token_lists:
        counter.update(tokens)

    # 1. Start with words that meet the frequency requirement
    vocab = [w for w, c in counter.most_common() if c >= min_freq]

    # 2. CRITICAL FIX: Manually insert "<UNK>" at index 0
    vocab = ["<UNK>"] + vocab

    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    return word2idx, idx2word


def tokens_to_indices(tokens, word2idx):
    return [word2idx.get(t, word2idx["<UNK>"]) for t in tokens]


def generate_skipgram_pairs(token_indices, window_size=2):
    """
    token_indices: list of word indices
    returns: list of (center, context) tuples
    """
    pairs = []
    n = len(token_indices)

    for i in range(n):
        center = token_indices[i]

        for j in range(max(0, i - window_size),
                       min(n, i + window_size + 1)):
            if i != j:
                context = token_indices[j]
                pairs.append((center, context))

    return pairs



def build_negative_sampling_table(counter, word2idx):
    vocab_size = len(word2idx)
    table = np.zeros(vocab_size)

    for word, idx in word2idx.items():
        table[idx] = counter.get(word, 0) ** 0.75

    table = table / table.sum()
    return table



def sample_negatives(prob_dist, num_samples):
    return np.random.choice(
        len(prob_dist),
        size=num_samples,
        replace=False,
        p=prob_dist
    )
