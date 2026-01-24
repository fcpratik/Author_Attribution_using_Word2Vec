from collections import Counter

import torch

from src.tokenizer import tokenize
from src.vocabulary import build_vocab, tokens_to_indices, build_negative_sampling_table,generate_skipgram_pairs
from src.word2vec import SkipGramSampling, train_word2vec

# -------- 1. Load text --------
with open("data/train_data/author_001.txt", "r", encoding="utf-8") as f:
    text = f.read()

# -------- 2. Tokenize --------
tokens = tokenize(text)

# -------- 3. Build vocab --------
counter = Counter(tokens)
word2idx, idx2word = build_vocab([tokens], min_freq=1)

# -------- 4. Tokens → indices --------
token_indices = tokens_to_indices(tokens, word2idx)

# -------- 5. Generate skip-gram pairs --------
pairs = generate_skipgram_pairs(token_indices, window_size=3)

# -------- 6. Negative sampling table --------
neg_dist = build_negative_sampling_table(counter, word2idx)

# -------- 7. Create model --------
model = SkipGramSampling(
    vocab_size=len(word2idx),
    embedding_dim=100
)

# -------- 8. Train --------
train_word2vec(
    model=model,
    pairs=pairs,
    neg_sampling_dist=neg_dist,
    epochs=5
)

# -------- 9. Save model --------
torch.save(model.state_dict(), "word2vec.pt")
