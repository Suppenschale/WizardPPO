import torch.nn as nn
import torch.nn.functional as F
import yaml


class PPONetwork(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        with open("parameter.yaml", "r") as f:
            config = yaml.safe_load(f)

        self.input_layer = config["nn"]["input_layer"]
        self.hidden_layer_shared = config["nn"]["hidden_layer_shared"]
        self.hidden_layer_each = config["nn"]["hidden_layer_each"]

        self.shared = nn.Linear(self.input_layer, self.hidden_layer_shared)
        self.hidden_layer_state = nn.Linear(self.hidden_layer_shared, self.hidden_layer_each)
        self.hidden_layer_action = nn.Linear(self.hidden_layer_shared, self.hidden_layer_each)
        self.hidden_layer_all_actions = nn.Linear(self.hidden_layer_each, 60)

        self.output_layer_value = nn.Linear(self.hidden_layer_each, 1)

    def forward(self, x, mask):

        shared = self.shared(x)
        shared = F.relu(shared)

        state = self.hidden_layer_state(shared)
        state = F.relu(state)
        state = self.output_layer_value(state)

        action = self.hidden_layer_action(shared)
        action = F.relu(action)
        action = self.hidden_layer_all_actions(action)
        action = action.masked_fill(~mask.bool(), float('-inf'))
        action = F.softmax(action)

        return state, action

