import torch
import torch.nn.functional as F
import yaml
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from tqdm import tqdm, trange

from env.bidding_heuristic import bidding_heuristic
from nn.PPO_network import PPONetwork
from env.environment import Environment


class Training:

    def __init__(self, policy: PPONetwork, optimizer: Optimizer):
        with open("parameter.yaml", "r") as f:
            config = yaml.safe_load(f)

        self.optimizer = optimizer
        self.game_iterations = config["train"]["game_iterations"]
        self.gamma = config["train"]["gamma"]
        self.lamb = config["train"]["lambda"]
        self.epochs = config["train"]["epochs"]
        self.minibatch_size = config["train"]["minibatch_size"]
        self.clip_eps = config["train"]["clip_eps"]
        self.a1 = config["train"]["a1"]
        self.a2 = config["train"]["a2"]
        self.num_players = config["env"]["num_players"]

        self.policies = [PPONetwork() for _ in range(self.num_players)]

        self.player_learning = 0
        self.policies[self.player_learning] = policy

    def collect_batch(self):

        states, action_masks, actions, rewards, log_probs, values = [], [], [], [], [], []

        for _ in range(self.game_iterations):
            env = Environment()
            for T in range(1, 60 // self.num_players + 1):
                env.start_round(T)

                for player in range(env.num_players):
                    env.bid(bidding_heuristic(env.players_hand[player], env.trump))

                for _ in range(T):

                    start = env.get_start_player()

                    for i in range(env.num_players):
                        state = env.get_state_vector()
                        action_mask = env.get_action_mask()

                        player = (start + i) % self.num_players

                        action, log_prob, value = self.policies[player].select_action(state, action_mask)

                        if player == self.player_learning:
                            states.append(state.detach())
                            action_masks.append(action_mask)
                            actions.append(action)
                            log_probs.append(log_prob.detach())
                            values.append(value.detach())

                        env.step(action)

                for _ in range(T):
                    rewards.append(env.players_points[self.player_learning])

        return {
            'states': torch.stack(states),
            'action_masks': torch.stack(action_masks),
            'actions': torch.tensor(actions, dtype=torch.long),
            'rewards': torch.tensor(rewards, dtype=torch.long),
            'log_probs': torch.stack(log_probs),
            'values': torch.stack(values).squeeze()
        }

    def compute_advantages(self, rewards, values):

        returns = np.zeros(len(rewards))
        advantages = np.zeros(len(rewards))

        for i in range(self.game_iterations):
            for player in reversed(range(self.num_players)):
                gae = 0
                value_next = 0
                for j in reversed(range(self.T)):
                    t = (i * self.T + j) * self.num_players + player

                    delta = rewards[t] + self.gamma * value_next - values[t]
                    gae = delta + self.gamma * self.lamb * gae
                    advantages[t] = gae
                    value_next = values[t]
                    returns[t] = gae + values[t]

        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    def ppo_update(self, batch):

        states = batch['states']
        action_masks = batch['action_masks']
        actions = batch['actions']
        log_probs_old = batch['log_probs']
        values = batch['values']
        rewards = batch['rewards']

        advantages, returns = self.compute_advantages(rewards, values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-12)

        dataset = torch.utils.data.TensorDataset(states, action_masks, actions, log_probs_old, returns, advantages)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)

        value_loss, loss = 0, 0

        for _ in range(self.epochs):
            for mb_states, mb_action_masks, mb_actions, mb_log_probs_old, mb_returns, mb_advantages in loader:
                mb_states = mb_states
                mb_action_masks = mb_action_masks
                mb_actions = mb_actions
                mb_log_probs_old = mb_log_probs_old
                mb_returns = mb_returns
                mb_advantages = mb_advantages

                value_pred, probs = self.policies[self.player_learning](mb_states, mb_action_masks)
                dist = torch.distributions.Categorical(probs)
                entropy = dist.entropy().mean()
                log_probs_new = dist.log_prob(mb_actions)

                ratio = torch.exp(log_probs_new - mb_log_probs_old)
                surrogate1 = ratio * mb_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                value_loss = F.mse_loss(value_pred.squeeze(), mb_returns)

                loss = policy_loss + self.a1 * value_loss - self.a2 * entropy

                #print(f"Value loss : {value_loss}, Loss : {loss}")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return value_loss.item(), loss.item()

    def training_loop(self, iterations):

        print("Start train loop!")
        batch = self.collect_batch()
        print(f"{batch['states'].shape=}")
        print("End train loop!")

        # value_losses = []
        # losses = []

        # pbar = trange(iterations)
        # for _ in pbar:
        #    batch = self.collect_batch()
        #    value_loss, loss = self.ppo_update(batch)
        #    pbar.set_postfix({
        #        "loss": f"{loss:.4f}",
        #        "value_loss": f"{value_loss:.4f}"
        #    })
        #    value_losses.append(value_loss)
        #    losses.append(loss)

        # x = range(1, iterations+1)
        # fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # (rows, columns)

        # First plot (on the left)
        # axs[0].plot(x, losses)
        # axs[0].set_title("Losses")

        # Second plot (on the right)
        # axs[1].plot(x, value_losses)
        # axs[1].set_title("Value Losses")

        # Improve layout
        # plt.tight_layout()
        # plt.show()
