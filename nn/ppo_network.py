import torch.distributions
import torch.nn as nn
import torch.nn.functional as F
import yaml

from nn.card_embedding import CardEmbedding
from nn.card_set_encoder import CardSetEncoder


class PPONetwork(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        with open("parameter.yaml", "r") as f:
            config = yaml.safe_load(f)

        self.num_players = config["env"]["num_players"]
        self.deck_size = config["env"]["deck_size"]
        self.emb_dim = config["embedding"]["emb_dim"]
        self.input_layer = config["nn"]["input_layer"]
        self.hidden_layer_shared = config["nn"]["hidden_layer_shared"]
        self.hidden_layer_each = config["nn"]["hidden_layer_each"]

        self.max_rounds = self.deck_size // self.num_players

        self.card_embedding = CardEmbedding(self.emb_dim)
        self.hand_encoder = CardSetEncoder(self.emb_dim)
        self.trick_encoder = CardSetEncoder(self.emb_dim)
        self.history_encoder = CardSetEncoder(self.emb_dim)

        self.shared = nn.Linear(self.input_layer, self.hidden_layer_shared)
        self.hidden_layer_state = nn.Linear(self.hidden_layer_shared, self.hidden_layer_each)
        self.hidden_layer_action = nn.Linear(self.hidden_layer_shared, self.hidden_layer_each)
        self.hidden_layer_all_actions = nn.Linear(self.hidden_layer_each, 54)  # 13 * 4 cards + 2 special

        self.output_layer_value = nn.Linear(self.hidden_layer_each, 1)

    def forward(self, x: torch.tensor, mask: torch.tensor):
        # Dims : [batch, 1]

        hand = x[:, :30]  # 15 cards each 2
        trick = x[:, 30:38]  # 4 cards each 2
        history = x[:, 38:158]  # 60 cards each 2
        trump_color = x[:, 158:159]  # 1 card each 2
        meta_info = x[:, 159:]  # rest

        hand_embedding, hand_mask = self.embed_cards(hand)
        trick_embedding, trick_mask = self.embed_cards(trick)
        history_embedding, history_mask = self.embed_cards(history)

        trump_color_embedding = self.card_embedding.suit_emb(trump_color - 1)

        hand_att = self.hand_encoder(hand_embedding, hand_mask)
        trick_att = self.trick_encoder(trick_embedding, trick_mask)
        history_att = self.trick_encoder(history_embedding, history_mask)

        shared = torch.cat([                          # (batch_size, 1213)
            hand_att.flatten(start_dim=1),                  # (batch_size, 15 * emb_dim)    = (batch_size, 225)
            trick_att.flatten(start_dim=1),                 # (batch_size,  4 * emb_dim)    = (batch_size,  60)
            history_att.flatten(start_dim=1),               # (batch_size, 60 * emb_dim)    = (batch_size, 900)
            trump_color_embedding.flatten(start_dim=1),     # (batch_size,  1 * emb_dim)    = (batch_size,  15)
            meta_info                                       # (batch_size, 13)              = (batch_size,  13)
        ], dim=1)

        shared = self.shared(shared)
        shared = F.relu(shared)

        value = self.hidden_layer_state(shared)
        value = F.relu(value)
        value = self.output_layer_value(value)

        probs = self.hidden_layer_action(shared)
        probs = F.relu(probs)
        probs = self.hidden_layer_all_actions(probs)
        probs = probs.masked_fill(~mask.bool(), float('-inf'))
        probs = F.softmax(probs, dim=-1)

        return value, probs

    def embed_cards(self, cards) -> torch.tensor:
        batch_size = cards.shape[0]
        max_size = cards.shape[1] // 2

        cards = cards.view(batch_size, max_size, 2)
        ranks = cards[..., 0]
        suits = cards[..., 1]

        masks = suits != 0

        embeddings = torch.zeros(batch_size, max_size, self.emb_dim)
        embeddings[masks] = self.card_embedding(ranks[masks], suits[masks])

        return embeddings, masks

    def select_action(self, x, mask):
        value, probs = self.forward(x, mask)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value
