import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

from src.vocabulary import sample_negatives_batch


class SkipGramDataset(Dataset):

    def __init__(self, pairs_array):
        self.centers = torch.from_numpy(pairs_array[:, 0].astype(np.int64))
        self.contexts = torch.from_numpy(pairs_array[:, 1].astype(np.int64))

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        return self.centers[idx], self.contexts[idx]


class SkipGramSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

        nn.init.uniform_(self.in_embed.weight, -0.5 / embedding_dim, 0.5 / embedding_dim)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, center_words, pos_context, neg_context):
        v = self.in_embed(center_words)          # (B, D)
        u_pos = self.out_embed(pos_context)      # (B, D)
        u_neg = self.out_embed(neg_context)      # (B, K, D)

        pos_score = torch.sum(v * u_pos, dim=1)
        pos_loss = F.logsigmoid(pos_score)

        neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)
        neg_loss = F.logsigmoid(-neg_score).sum(1)

        loss = -(pos_loss + neg_loss).mean()
        return loss


def train_word2vec(
    model,
    pairs_array,
    neg_table,
    num_negative=5,
    epochs=5,
    batch_size=4096,
    lr=0.003,
    num_workers=0,
):
    print("Training on: CPU")

    dataset = SkipGramDataset(pairs_array)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        t0 = time.time()

        for centers, pos in loader:
            bs = centers.size(0)

            # Batch negative sampling — one call for the whole batch
            neg_np = sample_negatives_batch(neg_table, bs, num_negative)
            neg = torch.from_numpy(neg_np.astype(np.int64))

            optimizer.zero_grad()
            loss = model(centers, pos, neg)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss:.4f}  Time: {elapsed:.1f}s")

    return model


def get_word_embeddings(model):
    return model.in_embed.weight.data