import torch
import torch.nn as nn


class CardSetEncoder(nn.Module):

    def __init__(self, emb_dim: int) -> None:
        super().__init__()

        self.attention = nn.MultiheadAttention(emb_dim, 3, batch_first=True)

    def forward(self, card_embeddings: torch.tensor, mask: torch.tensor) -> torch.tensor:

        attended, _ = self.attention(card_embeddings, card_embeddings, card_embeddings,
                                     key_padding_mask=~mask)

        return attended
