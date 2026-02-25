import torch.distributions
import torch.nn as nn
import torch.nn.functional as F
import yaml


class PPONetwork(nn.Module):

    def __init__(self, path="parameter.yaml") -> None:
        super().__init__()

        with open(path, "r") as f:
            config = yaml.safe_load(f)

        self.num_players = config["env"]["num_players"]
        self.deck_size = config["env"]["deck_size"]
        self.input_layer = config["nn"]["input_layer"]
        self.hidden_layer_shared = config["nn"]["hidden_layer_shared"]
        self.hidden_layer_each = config["nn"]["hidden_layer_each"]

        self.shared = nn.Linear(self.input_layer, self.hidden_layer_shared)
        self.hidden_layer_state = nn.Linear(self.hidden_layer_shared, self.hidden_layer_each)
        self.hidden_layer_action = nn.Linear(self.hidden_layer_shared, self.hidden_layer_each)
        self.hidden_layer_all_actions = nn.Linear(self.hidden_layer_each, 54)  # 13 * 4 cards + 2 special

        self.output_layer_value = nn.Linear(self.hidden_layer_each, 1)

    def forward(self, x: torch.tensor, mask: torch.tensor):
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

    def select_action(self, x, mask):
        with torch.no_grad():
            value, logits = self.forward(x, mask)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def select_action_greedy(self, x, mask):
        with torch.no_grad():
            value, logits = self.forward(x, mask)
            action = torch.argmax(logits, dim=1)
            log_prob = torch.log_softmax(logits, dim=1).gather(1, action.unsqueeze(1)).squeeze(1)
        return action.item(), log_prob, value
