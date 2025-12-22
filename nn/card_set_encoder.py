import torch
import torch.nn as nn


class CardSetEncoder(nn.Module):

    def __init__(self, emb_dim: int) -> None:
        super().__init__()

        self.attention = nn.MultiheadAttention(emb_dim, 3, batch_first=True)
        self.empty_hand = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.emb_dim = emb_dim

    def forward(self, card_embeddings: torch.tensor, mask: torch.tensor) -> torch.tensor:

        all_masked = ~mask.any(dim=1)

        if all_masked.any():
            attended = torch.zeros_like(card_embeddings)
            non_empty = ~all_masked
            if non_empty.any():
                attended[non_empty], _ = self.attention(
                    card_embeddings[non_empty],
                    card_embeddings[non_empty],
                    card_embeddings[non_empty],
                    key_padding_mask=~mask[non_empty])

            attended[all_masked] = self.empty_hand.expand(
                all_masked.sum(), card_embeddings.size(1), self.emb_dim)

            return attended

        attended, _ = self.attention(
            card_embeddings, card_embeddings, card_embeddings,
            key_padding_mask=~mask)
        return attended
