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
        #trump_color = x[:, 158:159].type(torch.long)    # 1 card each 2
        #meta_info = x[:, 159:]                          # rest

        #print(f"{hand=}")
        #print(f"{trick=}")
        #print(f"{history=}")
        #print(f"{trump_color=}")
        #print(f"{meta_info=}")

        #hand_embedding, hand_mask = self.embed_cards(hand)
        #trick_embedding, trick_mask = self.embed_cards(trick)
        #history_embedding, history_mask = self.embed_cards(history)

        #trump_color_embedding = self.card_embedding(torch.zeros_like(trump_color), trump_color, trump=True)

        #shared = torch.cat([                         # (batch_size, 1213)
        #    hand_embedding.flatten(start_dim=1),            # (batch_size,   15 * emb_dim)  = (batch_size, 225)
        #    trick_embedding.flatten(start_dim=1),           # (batch_size,    4 * emb_dim)  = (batch_size,  60)
        #    history_embedding.flatten(start_dim=1),         # (batch_size,   60 * emb_dim)  = (batch_size, 900)
        #    trump_color_embedding.flatten(start_dim=1),     # (batch_size,    1 * emb_dim)  = (batch_size,  15)
        #    meta_info                                       # (batch_size,    3)            = (batch_size,  13)
        #], dim=1)

        #for i in range(shared.shape[1]):  # 1213 elements
        #    print(f"[{i}] = {shared[0, i].item():.8f}")

        one_hot = torch.zeros(x.shape[0], 3)

        mask0 = x[:, 0] <= 0.05
        mask1 = (x[:, 0] > 0.05) & (x[:, 0] <= 0.1)
        mask2 = x[:, 0] > 0.1

        one_hot[mask0, 0] = 1.0
        one_hot[mask1, 1] = 1.0
        one_hot[mask2, 2] = 1.0

        shared = self.shared(one_hot)
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

    def print_special_state(self, i, x, mask):
        _, logits = self.forward(x, mask)
        prob = F.softmax(logits, dim=1)
        print(
            f"Probability to pick move for {i} tricks : Red 10 : {prob[..., 9].item()}, Yellow 13 : {prob[..., 25].item()} ({x})")

    def debug_simple_network(self, x: torch.tensor, mask: torch.tensor):
        print(f"\n=== SIMPLE NETWORK DEBUG ===")

        # Extract only tricks_left (position 169)
        tricks_left_idx = 169  # Based on your state structure
        tricks_left = x

        print(f"Input shape: {x.shape}")
        print(f"tricks_left value: {tricks_left.item():.6f}")
        print(f"tricks_left range expected: [0, 1]")

        # Forward pass step by step
        shared = self.shared(x)
        print(f"\nAfter shared linear:")
        print(f"  Shape: {shared.shape}")
        print(f"  Values: {shared[0].detach().cpu().numpy()}")
        print(f"  Min: {shared.min().item():.6f}, Max: {shared.max().item():.6f}")

        shared_relu = F.relu(shared)
        print(f"\nAfter ReLU:")
        print(f"  Values: {shared_relu[0].detach().cpu().numpy()}")
        print(f"  Zeros: {(shared_relu == 0).sum().item()}/{shared_relu.numel()}")

        value_path = self.hidden_layer_state(shared_relu)
        value_relu = F.relu(value_path)
        value_out = self.output_layer_value(value_relu)

        print(f"\nValue path:")
        print(f"  After hidden_state: {value_path[0].detach().cpu().numpy()}")
        print(f"  After ReLU: {value_relu[0].detach().cpu().numpy()}")
        print(f"  Final value: {value_out.item():.6f}")

        # Check gradients
        loss = value_out.mean()
        loss.backward()

        grad_sum = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_sum += grad_norm
                print(f"  {name} gradient norm: {grad_norm:.6f}")

        print(f"\nTotal gradient norm: {grad_sum:.6f}")

        return value_out
