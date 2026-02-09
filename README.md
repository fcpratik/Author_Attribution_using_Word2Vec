# Word2Vec Author Attribution

A from-scratch implementation of Word2Vec using PyTorch for author attribution tasks. This project implements the Skip-Gram model with negative sampling to learn word embeddings, which are then used for author verification and clustering.

## 📋 Overview

This implementation supports two main tasks:
- **Task 1: Author Verification** - Determine if two text chunks were written by the same author
- **Task 2: Author Clustering** - Group text chunks by author using unsupervised clustering

## 🏗️ Project Structure

```
.
├── data/
│   └── train_data/          # Training data (author_001.txt, author_002.txt, ...)
├── sample_inputs/           # Example test inputs
├── src/
│   ├── tokenizer.py         # Text tokenization utilities
│   ├── vocabulary.py        # Vocabulary building and skip-gram generation
│   └── word2vec.py          # Skip-Gram model implementation
├── hungarian_eval.py        # Task 2 evaluation using Hungarian algorithm
├── main.py                  # Example training script
├── run_model.sh            # Main entry point for training/testing
├── install_requirements.sh  # Package installation script
└── README.md               # This file
```

## 🚀 Quick Start

### Installation

```bash
# Install required packages
bash install_requirements.sh

# Or manually install
pip install torch numpy scipy
```

### Training

```bash
# Train on the provided training data
./run_model.sh data/train_data/
```

### Testing

```bash
# Task 1: Author Verification
./run_model.sh test1 sample_inputs/sample_task1.json output/

# Task 2: Author Clustering
./run_model.sh test2 sample_inputs/sample_task2.json output/
```

## 🧩 Implementation Details

### Tokenization (`src/tokenizer.py`)

- Converts text to lowercase
- Tokenizes using regex pattern:
  - Words with optional apostrophes (e.g., "don't", "I'm")
  - Numbers
  - Punctuation as separate tokens

### Vocabulary (`src/vocabulary.py`)

**Features:**
- `build_vocab()` - Builds vocabulary with frequency thresholding
  - Includes special `<UNK>` token at index 0 for out-of-vocabulary words
  - Only includes words meeting minimum frequency requirement
- `generate_skipgram_pairs()` - Creates (center, context) word pairs for training
  - Configurable window size (default: 2)
- `build_negative_sampling_table()` - Creates probability distribution for negative sampling
  - Uses subsampling with power 0.75 (as per original Word2Vec paper)
- `sample_negatives()` - Samples negative examples during training

### Word2Vec Model (`src/word2vec.py`)

**Architecture:**
- **Skip-Gram with Negative Sampling (SGNS)**
- Two embedding matrices:
  - `in_embed` - Input/center word embeddings
  - `out_embed` - Output/context word embeddings
- Initialization:
  - Input embeddings: uniform distribution [-0.5/dim, 0.5/dim]
  - Output embeddings: zeros

**Training:**
- Optimizer: Adam
- Loss: Negative sampling loss using log-sigmoid
- Configurable hyperparameters:
  - `embedding_dim` - Dimensionality of word vectors (default: 100)
  - `num_negative` - Number of negative samples per positive pair (default: 5)
  - `batch_size` - Training batch size (default: 512)
  - `lr` - Learning rate (default: 0.003)
  - `epochs` - Number of training epochs (default: 5)

### Evaluation (`hungarian_eval.py`)

For Task 2 (clustering), the Hungarian algorithm is used to find the optimal mapping between predicted clusters and true author labels, maximizing clustering accuracy.

## 📊 Example Usage

```python
from collections import Counter
import torch
from src.tokenizer import tokenize
from src.vocabulary import build_vocab, tokens_to_indices, build_negative_sampling_table, generate_skipgram_pairs
from src.word2vec import SkipGramSampling, train_word2vec

# Load and tokenize text
with open("data/train_data/author_001.txt", "r") as f:
    text = f.read()
tokens = tokenize(text)

# Build vocabulary
counter = Counter(tokens)
word2idx, idx2word = build_vocab([tokens], min_freq=1)

# Convert tokens to indices
token_indices = tokens_to_indices(tokens, word2idx)

# Generate training pairs
pairs = generate_skipgram_pairs(token_indices, window_size=3)

# Create negative sampling distribution
neg_dist = build_negative_sampling_table(counter, word2idx)

# Initialize and train model
model = SkipGramSampling(vocab_size=len(word2idx), embedding_dim=100)
train_word2vec(model, pairs, neg_dist, epochs=5)

# Save trained model
torch.save(model.state_dict(), "word2vec.pt")

# Get word embeddings
embeddings = model.in_embed.weight.data
```

## 📝 Important Notes

- **No Pre-trained Models**: You must implement and train Word2Vec from scratch (no gensim or pre-trained embeddings)
- **Anonymized Authors**: Training data uses anonymized author IDs (author_001, author_002, etc.)
- **Variable Chunk Sizes**: Text chunks range from 50-500 words
- **Unseen Authors**: Test data may contain authors not present in training data
- **Transductive Learning**: For Task 2, you can fine-tune embeddings on test data (since it's unsupervised clustering)

## 🔧 Customization

To modify the model for your needs:

1. **Adjust hyperparameters** in `src/word2vec.py`:
   - Embedding dimension
   - Learning rate
   - Number of negative samples
   - Window size

2. **Extend tokenization** in `src/tokenizer.py`:
   - Add custom regex patterns
   - Include preprocessing steps

3. **Implement inference scripts**:
   - `src/inference_task1.py` for author verification
   - `src/inference_task2.py` for author clustering

