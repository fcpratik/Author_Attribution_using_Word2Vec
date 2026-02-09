import sys
import os
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import time

from src.tokenizer import tokenize

from src.vocabulary import (
    build_vocab, tokens_to_indices,
    build_negative_sampling_table, generate_skipgram_pairs_array,
    subsample_tokens,
)

from src.word2vec import SkipGramSampling, train_word2vec


def load_training_data(train_dir):
    print(f"Loading text data from: {train_dir}")
    all_tokens_list = []
    txt_files = sorted(Path(train_dir).glob("*.txt"))
    print(f"Found {len(txt_files)} text files")

    for filepath in txt_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            tokens = tokenize(text)
            all_tokens_list.append(tokens)
            print(f"  {filepath.name}: {len(tokens):,} tokens")

    return all_tokens_list


def main():
    TRAIN_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/train_data"
    EMBEDDING_DIM = 150
    WINDOW_SIZE = 5
    MIN_FREQ = 2
    NUM_NEGATIVE = 10
    EPOCHS = 15
    BATCH_SIZE = 4096
    LEARNING_RATE = 0.003
    SUBSAMPLE_THRESH = 1e-4

    print("=" * 60)
    print("OPTIMIZED Word2Vec Training")
    print("=" * 60)
    print(f"Embedding dim: {EMBEDDING_DIM}")
    print(f"Window size:   {WINDOW_SIZE}")
    print(f"Epochs:        {EPOCHS}")
    print(f"Batch size:    {BATCH_SIZE}")
    print(f"Subsampling:   {SUBSAMPLE_THRESH}")
    print("=" * 60)

    start_time = time.time()

    # ---- Load the data ----
    all_tokens_list = load_training_data(TRAIN_DIR)

    # ---- Build vocabulary on all data ----
    print("\nBuilding vocabulary...")
    counter = Counter()
    for tokens in all_tokens_list:
        counter.update(tokens)

    word2idx, idx2word = build_vocab(counter, min_freq=MIN_FREQ)

    # Total length of vocabulary
    vocab_size = len(word2idx)

    total_tokens = sum(len(t) for t in all_tokens_list)
    print(f"Total tokens:    {total_tokens:,}")
    print(f"Vocabulary size: {vocab_size:,}")

    # ---- Generate skip-gram pairs  ----
    print("\nGenerating skip-gram pairs ...")
    pair_chunks = []
    total_after_subsample = 0

    for tokens in all_tokens_list:
        indices = tokens_to_indices(tokens, word2idx)
        indices = subsample_tokens(indices, counter, word2idx, threshold=SUBSAMPLE_THRESH)
        total_after_subsample += len(indices)
        pairs = generate_skipgram_pairs_array(indices, window_size=WINDOW_SIZE)
        pair_chunks.append(pairs)


    all_pairs = np.concatenate(pair_chunks, axis=0)

    print(f"Tokens after subsampling: {total_after_subsample:,}")
    print(f"Generated {len(all_pairs):,} training pairs")

    # ---- Negative sampling table ----
    neg_table = build_negative_sampling_table(counter, word2idx)

    # ---- Create a model ----
    print(f"\nCreating model ({vocab_size:,} x {EMBEDDING_DIM})...")
    model = SkipGramSampling(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ---- Training part ----
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    model = train_word2vec(
        model=model,
        pairs_array=all_pairs,
        neg_table=neg_table,
        num_negative=NUM_NEGATIVE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        num_workers=0,
    )

    # ---- Saving  ----
    print("\nSaving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'word2idx': word2idx,
        'idx2word': idx2word,
        'embedding_dim': EMBEDDING_DIM,
        'vocab_size': vocab_size,
        'hyperparameters': {
            'window_size': WINDOW_SIZE,
            'min_freq': MIN_FREQ,
            'num_negative': NUM_NEGATIVE,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'lr': LEARNING_RATE,
        }
    }, 'word2vec_model.pt')

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training complete in {elapsed / 60:.1f} minutes!")
    print(f"Model saved to 'word2vec_model.pt'")
    print("=" * 60)


if __name__ == '__main__':
    main()