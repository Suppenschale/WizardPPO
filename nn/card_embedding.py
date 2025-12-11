import torch
import torch.nn as nn
import yaml

from env.card import Card


class CardEmbedding(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        with open("parameter.yaml", "r") as f:
            config = yaml.safe_load(f)

        self.emb_dim = config["embedding"]["emb_dim"]

        self.rank_emb = nn.Embedding(15, self.emb_dim)
        self.suit_emb = nn.Embedding(5, self.emb_dim)

    def forward(self, card: Card):
        rank_id = torch.tensor([card.rank], dtype=torch.long)
        suit_id = torch.tensor([card.suit.value - 1], dtype=torch.long)

        r = self.rank_emb(rank_id)
        s = self.suit_emb(suit_id)
        return r + s
