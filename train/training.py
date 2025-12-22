import concurrent.futures
import multiprocessing
import time
from functools import partial

import torch
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter
import yaml
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from tqdm import tqdm, trange

from env.bidding_heuristic import bidding_heuristic
from nn.ppo_network import PPONetwork
from env.environment import Environment


def collect_batch(game_id, policies, player_learning):
    import torch
    from env.environment import Environment

    print(f"Start game {game_id}")

    states_local, action_masks_local, actions_local, rewards_local, log_probs_local, values_local \
        = [], [], [], [], [], []

    env = Environment()
    env.start_game()
    for cur_round in range(env.max_rounds):

        for player in range(env.num_players):
            env.bid(bidding_heuristic(env.players_hand[player], env.trump))

        for _ in range(env.num_rounds):

            start = env.get_start_player()
            for i in range(env.num_players):
                state = env.get_state_vector()
                action_mask = env.get_action_mask()

                player = (start + i) % env.num_players

                with torch.no_grad():
                    action, log_prob, value = policies[player].select_action(state, action_mask)

                if player == player_learning:
                    states_local.append(state.detach())
                    action_masks_local.append(action_mask)
                    actions_local.append(action)
                    log_probs_local.append(log_prob.detach())
                    values_local.append(value.detach())

                env.step(action)

        for _ in range(cur_round):
            rewards_local.append(0)
        rewards_local.append(env.players_round_points_history[cur_round][player_learning])

    rewards_local[-1] = env.players_game_points[0]

    print(f"End game {game_id}")

    return {
        'states': states_local,
        'action_masks': action_masks_local,
        'actions': actions_local,
        'rewards': rewards_local,
        'log_probs': log_probs_local,
        'values': values_local
    }


class Training:

    def __init__(self, policy: PPONetwork, optimizer: Optimizer, path: str):
        with open("parameter.yaml", "r") as f:
            config = yaml.safe_load(f)

        self.policy = policy
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
        self.deck_size = config["env"]["deck_size"]
        self.game_length = self.deck_size // self.num_players
        self.num_of_rounds = (self.game_length * (self.game_length + 1)) // 2

        self.policies = [PPONetwork() for _ in range(self.num_players)]

        self.path = path
        self.writer = SummaryWriter(log_dir=os.path.join(path, "board"))
        self.step = 0

        self.player_learning = 0
        self.policies[self.player_learning] = policy

    def collect_batch_parallel(self):

        states, action_masks, actions, rewards, log_probs, values = [], [], [], [], [], []

        worker_func = partial(collect_batch, policies=self.policies, player_learning=self.player_learning)

        num_workers = min(self.game_iterations, multiprocessing.cpu_count())
        print(f"{num_workers=}")

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executors:
            future_to_game = {executors.submit(worker_func, game): game for game in range(self.game_iterations)}

            for future in concurrent.futures.as_completed(future_to_game):
                try:
                    result = future.result()
                    states.extend(result['states'])
                    action_masks.extend(result['action_masks'])
                    actions.extend(result['actions'])
                    rewards.extend(result['rewards'])
                    log_probs.extend(result['log_probs'])
                    values.extend(result['values'])

                except Exception as e:
                    print(f"Game {future_to_game[future]} generated an exception: {e}")

        return {
            'states': torch.cat(states),
            'action_masks': torch.cat(action_masks),
            'actions': torch.tensor(actions),
            'rewards': torch.tensor(rewards),
            'log_probs': torch.cat(log_probs),
            'values': torch.cat(values).squeeze(-1)
        }

    def compute_advantages(self, rewards, values):

        returns = [0 for _ in range(len(rewards))]
        advantages = [0 for _ in range(len(rewards))]
        T = self.num_of_rounds

        for n in range(self.game_iterations):
            gae = 0
            value_next = 0
            for i in reversed(range(T)):
                t = n * T + i
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
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.minibatch_size, shuffle=False)

        value_loss, loss = 0, 0

        # torch.autograd.set_detect_anomaly(True)

        for _ in range(self.epochs):
            for mb_states, mb_action_masks, mb_actions, mb_log_probs_old, mb_returns, mb_advantages in loader:
                value, probs = self.policy(mb_states, mb_action_masks)

                dist = torch.distributions.Categorical(probs)
                entropy = dist.entropy().mean()
                log_probs_new = dist.log_prob(mb_actions)

                ratio = torch.exp(log_probs_new - mb_log_probs_old)
                surrogate1 = ratio * mb_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                value_loss = F.mse_loss(value.squeeze(), mb_returns)

                loss = policy_loss + self.a1 * value_loss - self.a2 * entropy

                #print(f"Value loss : {value_loss}, Loss : {loss}")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return value_loss.item(), loss.item()

    def training_loop(self, iterations):

        value_losses = []
        losses = []

        start_time = time.perf_counter()
        batch = self.collect_batch_parallel()
        print(f"{time.perf_counter()-start_time}")

        start_time = time.perf_counter()
        for game in range(self.game_iterations):
            collect_batch(game, self.policies, self.player_learning)
        print(f"{time.perf_counter()-start_time}")
        # value_loss, loss = self.ppo_update(batch)

        for k, v in batch.items():
            print(f"{k} : {v.shape}")

        return

        pbar = trange(iterations)
        for i in pbar:
            batch = self.collect_batch()
            value_loss, loss = self.ppo_update(batch)
            pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "value_loss": f"{value_loss:.4f}"
            })
            self.writer.add_scalar(tag="Loss/Value", scalar_value=value_loss, global_step=i)
            self.writer.add_scalar(tag="Loss/Policy", scalar_value=loss, global_step=i)
            self.writer.flush()
            value_losses.append(value_loss)
            losses.append(loss)

        x = range(1, iterations + 1)
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        # First plot (on the left)
        axs[0].plot(x, losses)
        axs[0].set_title("Losses")

        # Second plot (on the right)
        axs[1].plot(x, value_losses)
        axs[1].set_title("Value Losses")

        # Improve layout
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(self.path, "losses.png"))  # save as PNG
        plt.close()  # optional: closes the figure

        torch.save(self.policies[self.player_learning].state_dict(), os.path.join(self.path, "model.pth"))
        np.save(os.path.join(self.path, "value-losses.npy"), np.array([]))
        np.save(os.path.join(self.path, "losses.npy"), np.array([]))
