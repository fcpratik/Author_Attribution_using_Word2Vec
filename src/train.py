import sys
import os
from pathlib import Path
from collections import Counter
import torch

from src.tokenizer import tokenize
from src.vocabulary import build_vocab, tokens_to_indices, build_negative_sampling_table, generate_skipgram_pairs_array
from src.word2vec import SkipGramSampling, train_word2vec


def load_training_data(train_dir):

    print(f"Loading training data from {train_dir}...")
    all_texts = []
    all_tokens_list = []

    # Read all .txt files
    txt_files = list(Path(train_dir).glob("*.txt"))
    print(f"Found {len(txt_files)} author files")

    for filepath in txt_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            all_texts.append(text)
            tokens = tokenize(text)
            all_tokens_list.append(tokens)
            print(f"  {filepath.name}: {len(tokens)} tokens")

    return all_texts, all_tokens_list


def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py <train_dir>")
        sys.exit(1)

    train_dir = sys.argv[1]

    # Hyperparameters
    EMBEDDING_DIM = 100
    WINDOW_SIZE = 3
    MIN_FREQ = 2
    NUM_NEGATIVE = 5
    EPOCHS = 5
    BATCH_SIZE = 512
    LEARNING_RATE = 0.003

    print("=" * 60)
    print("Word2Vec Training for Author Attribution")
    print("=" * 60)

    # Load data
    all_texts, all_tokens_list = load_training_data(train_dir)

    # Build vocabulary on all data
    print("\nBuilding vocabulary...")
    counter = Counter()
    for tokens in all_tokens_list:
        counter.update(tokens)

    word2idx, idx2word = build_vocab(all_tokens_list, min_freq=MIN_FREQ)
    vocab_size = len(word2idx)
    print(f"Vocabulary size: {vocab_size} (min_freq={MIN_FREQ})")

    # Convert all tokens to indices
    print("\nConverting tokens to indices...")
    all_token_indices = []
    for tokens in all_tokens_list:
        indices = tokens_to_indices(tokens, word2idx)
        all_token_indices.extend(indices)

    print(f"Total tokens: {len(all_token_indices)}")

    # Generate skip-gram pairs
    print("\nGenerating skip-gram pairs...")
    # We need to process each document separately to avoid cross-document pairs
    all_pairs = []
    for tokens in all_tokens_list:
        indices = tokens_to_indices(tokens, word2idx)
        pairs = generate_skipgram_pairs_array(indices, window_size=WINDOW_SIZE)
        all_pairs.extend(pairs)

    print(f"Generated {len(all_pairs)} training pairs")

    # Build negative sampling distribution
    print("\nBuilding negative sampling table...")
    neg_dist = build_negative_sampling_table(counter, word2idx)

    # Create model
    print("\nInitializing model...")
    model = SkipGramSampling(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    train_word2vec(
        model=model,
        pairs_array=all_pairs,
        neg_table=neg_dist,
        num_negative=NUM_NEGATIVE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE
    )

    # Save model and vocabulary
    print("\nSaving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'word2idx': word2idx,
        'idx2word': idx2word,
        'embedding_dim': EMBEDDING_DIM,
        'vocab_size': vocab_size
    }, 'word2vec_model.pt')

    print("\n" + "=" * 60)
    print("Training complete! Model saved to word2vec_model.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
