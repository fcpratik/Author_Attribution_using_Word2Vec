import torch
import torch.nn as nn
import torch.nn.functional as F

from src.vocabulary import sample_negatives


class SkipGramSampling(nn.Module):
    def __init__(self,vocab_size,embedding_dim):
        super().__init__()

        self.in_embed = nn.Embedding(vocab_size,embedding_dim)
        self.out_embed = nn.Embedding(vocab_size,embedding_dim)

        nn.init.uniform(self.in_embed.weight,-0.5/embedding_dim,0.5/embedding_dim)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self,center_words,pos_context,neg_context):
        v=self.in_embed(center_words)
        u_pos=self.out_embed(pos_context)
        u_neg = self.out_embed(neg_context)

        pos_score = torch.sum(v*u_pos,dim=1)
        pos_loss = F.logsigmoid(pos_score)

        neg_score = torch.bmm(u_neg,v.unsqueeze(2)).squeeze(2)
        neg_loss = F.logsigmoid(-neg_score).sum(1)

        loss = -(pos_loss+neg_loss).mean()
        return loss


def get_batches(pairs,batch_size):
    for i in range(0,len(pairs),batch_size):
        yield pairs[i:i+batch_size]


def train_word2vec(
        model,
        pairs,
        neg_sampling_dist,
        num_negative = 5,
        epochs=5,
        batch_size=512,
        lr=0.003
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0

        for batch in get_batches(pairs,batch_size):
            centers = torch .tensor([c for c,_ in batch],dtype=torch.long)
            pos = torch.tensor([p for _,p in batch],dtype=torch.long)

            neg=torch.tensor(
                [sample_negatives(neg_sampling_dist,num_negative)
                 for _ in batch],
                dtype=torch.long
            )

            optimizer.zero_grad()
            loss = model(centers,pos,neg)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f} ")


def get_word_embeddings(model):
    return model.in_embed.weight.data




