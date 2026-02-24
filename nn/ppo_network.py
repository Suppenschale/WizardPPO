import torch.distributions
import torch.nn as nn
import torch.nn.functional as F
import yaml

from nn.card_embedding import CardEmbedding
from nn.card_set_encoder import CardSetEncoder


class PPONetwork(nn.Module):

    def __init__(self, path="parameter.yaml") -> None:
        super().__init__()

        with open(path, "r") as f:
            config = yaml.safe_load(f)

        self.num_players = config["env"]["num_players"]
        self.deck_size = config["env"]["deck_size"]
        self.emb_dim = config["embedding"]["emb_dim"]
        self.input_layer = config["nn"]["input_layer"]
        self.hidden_layer_shared = config["nn"]["hidden_layer_shared"]
        self.hidden_layer_each = config["nn"]["hidden_layer_each"]

        self.max_rounds = self.deck_size // self.num_players

        self.card_embedding = CardEmbedding(self.emb_dim)

        self.shared = nn.Linear(self.input_layer, self.hidden_layer_shared)
        self.hidden_layer_state = nn.Linear(self.hidden_layer_shared, self.hidden_layer_each)
        self.hidden_layer_action = nn.Linear(self.hidden_layer_shared, self.hidden_layer_each)
        self.hidden_layer_all_actions = nn.Linear(self.hidden_layer_each, 54)  # 13 * 4 cards + 2 special

        self.output_layer_value = nn.Linear(self.hidden_layer_each, 1)

    def forward(self, x: torch.tensor, mask: torch.tensor):
        # Dims : [batch, 1]

        #hand = x[:, :30].type(torch.long)               # 15 cards each 2
        #trick = x[:, 30:38].type(torch.long)            # 4 cards each 2
        #history = x[:, 38:158].type(torch.long)         # 60 cards each 2
        #trump_color = x[:, 158:159].type(torch.long)    # 1 color
        #meta_info = x[:, 159:]                          # rest

        #hand_embedding, hand_mask = self.embed_cards(hand)
        #trick_embedding, trick_mask = self.embed_cards(trick)
        #history_embedding, history_mask = self.embed_cards(history)

        #trump_color_embedding = self.card_embedding(torch.zeros_like(trump_color), trump_color, trump=True)

        #shared = torch.cat([                         # (batch_size, 1213)
        #    hand_embedding.flatten(start_dim=1),            # (batch_size,   15 * emb_dim)  = (batch_size, 225)
        #    trick_embedding.flatten(start_dim=1),           # (batch_size,    4 * emb_dim)  = (batch_size,  60)
        #    history_embedding.flatten(start_dim=1),         # (batch_size,   60 * emb_dim)  = (batch_size, 900)
        #    trump_color_embedding.flatten(start_dim=1),     # (batch_size,    1 * emb_dim)  = (batch_size,  15)
        #    meta_info                                       # (batch_size,  177)            = (batch_size,  13)
        #], dim=1)

        #for i in range(shared.shape[1]):  # 1213 elements
        #    print(f"[{i}] = {shared[0, i].item():.8f}")

        shared = self.shared(x)
        shared = F.relu(shared)

        value = self.hidden_layer_state(shared)
        value = F.relu(value)
        value = self.output_layer_value(value)

        probs = self.hidden_layer_action(shared)
        probs = F.relu(probs)
        logits = self.hidden_layer_all_actions(probs)
        logits = logits.masked_fill(~mask.bool(), float('-inf'))

        return value, logits

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
        with torch.no_grad():
            value, logits = self.forward(x, mask)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            prob = F.softmax(logits, dim=1)
            # print(f"Probability to pick move: Red 10 : {prob[..., 9]}, Yellow 13 : {prob[..., 25]}")
        return action.item(), dist.log_prob(action), value

    def select_action_greedy(self, x, mask):
        with torch.no_grad():
            value, logits = self.forward(x, mask)
            action = torch.argmax(logits, dim=1)
            log_prob = torch.log_softmax(logits, dim=1).gather(1, action.unsqueeze(1)).squeeze(1)

        return action.item(), log_prob, value

    def print_special_state(self, i, x, mask):
        _, logits = self.forward(x, mask)
        prob = F.softmax(logits, dim=1)
        print(
            f"Probability to pick move for {i} tricks : Red 10 : {prob[0][9].item()}, Yellow 13 : {prob[0][25].item()} ()")
        return prob[0][9].item(), prob[0][25].item()
