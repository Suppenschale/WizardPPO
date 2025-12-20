import torch
import torch.nn as nn
import yaml

from env.card import Card


class CardEmbedding(nn.Module):

    def __init__(self, emb_dim: int) -> None:
        super().__init__()

        self.emb_dim = emb_dim
        self.rank_emb = nn.Embedding(15, self.emb_dim)
        self.suit_emb = nn.Embedding(5, self.emb_dim)

    def forward(self, rank: torch.tensor, suit: torch.tensor):
        rank_id = rank
        suit_id = suit - 1

        r = self.rank_emb(rank_id)
        s = self.suit_emb(suit_id)
        return r + s
