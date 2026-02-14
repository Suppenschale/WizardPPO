import concurrent.futures
import copy
import itertools
import multiprocessing
import pickle
import random
import time
from functools import partial

import torch
import torch.nn.functional as F
import os

from tensorboard import program
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import yaml
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from tqdm import tqdm, trange

from env.bidding_heuristic import bidding_heuristic
from nn.ppo_network import PPONetwork
from env.environment import Environment
from sim.simulation import Simulation

FIXED_TEST_ROUND = 11

STATE_BID_0 = None
MASK_BID_0 = None
STATE_BID_1 = None
MASK_BID_1 = None
STATE_BID_2 = None
MASK_BID_2 = None


def collect_batch(game_id, policy, player_learning):
    import torch
    from env.environment import Environment

    env = Environment()
    cur_round = FIXED_TEST_ROUND
    env.start_round(cur_round, random.choice(range(env.num_players)))

    states_local        = [[] for _ in range(env.num_players)]
    action_masks_local  = [[] for _ in range(env.num_players)]
    actions_local       = [[] for _ in range(env.num_players)]
    rewards_local       = [[] for _ in range(env.num_players)]
    log_probs_local     = [[] for _ in range(env.num_players)]
    values_local        = [[] for _ in range(env.num_players)]

    for player in range(env.num_players):
        max_value = -10000000
        max_bid = -1
        for test_bid in range(env.num_rounds + 1):
            env.players_bid[env.cur_player] = test_bid
            state = env.get_state_vector()
            action_mask = env.get_action_mask()
            value, _ = policy(state, action_mask)
            if value > max_value:
                max_value = value
                max_bid = test_bid
        selected_bid = bidding_heuristic(cur_round, max_bid)
        env.bid(selected_bid)

    for _ in range(env.num_rounds):

        start = env.get_start_player()
        for i in range(env.num_players):
            state = env.get_state_vector()
            action_mask = env.get_action_mask()

            player = (start + i) % env.num_players

            with torch.no_grad():
                action, log_prob, value = policy.select_action(state, action_mask)

            states_local[player].append(state)
            action_masks_local[player].append(action_mask)
            actions_local[player].append(action)
            log_probs_local[player].append(log_prob)
            values_local[player].append(value)

            env.step(action)

    for player in range(env.num_players):
        for _ in range(cur_round - 1):
            rewards_local[player].append(0)
        reward = env.players_points[player]
        rewards_local[player].append(reward if True else reward * 10)

    return {
        'states': [state for state_list in states_local for state in state_list],
        'action_masks': [action_mask for action_mask_list in action_masks_local for action_mask in action_mask_list],
        'actions': [action for action_list in actions_local for action in action_list],
        'rewards': [reward for reward_list in rewards_local for reward in reward_list],
        'log_probs': [log_prob for log_prob_list in log_probs_local for log_prob in log_prob_list],
        'values': [value for value_list in values_local for value in value_list]
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

        #self.policies = [PPONetwork() for _ in range(self.num_players)]

        self.DEBUG_PRINT: bool = config["env"]["debug_print"]

        self.path = path
        self.writer = SummaryWriter(log_dir=os.path.join(path, "board"))
        self.step = 0

        self.player_learning = 0
        #self.policies[self.player_learning] = policy

        self.train_count = 0
        self.eval_count = 0
        self.best_model = copy.deepcopy(policy.state_dict())
        self.best_winrate = 0

    def collect_batch_parallel(self):

        states, action_masks, actions, rewards, log_probs, values = [], [], [], [], [], []

        worker_func = partial(collect_batch, policy=self.policy, player_learning=self.player_learning)

        num_workers = min(self.game_iterations, multiprocessing.cpu_count())

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executors:
            future_to_game = {executors.submit(worker_func, game): game for game in range(self.game_iterations)}

            for future in concurrent.futures.as_completed(future_to_game):
                #try:
                result = future.result()
                states.extend(result['states'])
                action_masks.extend(result['action_masks'])
                actions.extend(result['actions'])
                rewards.extend(result['rewards'])
                log_probs.extend(result['log_probs'])
                values.extend(result['values'])

            # except Exception as e:
            #    print(f"Game {future_to_game[future]} generated an exception: {e}")

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
        T = FIXED_TEST_ROUND  # self.num_of_rounds

        for n in range(self.game_iterations * self.num_players):
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
        log_probs_old = batch['log_probs'].detach()
        values = batch['values'].detach()
        rewards = batch['rewards']

        advantages, returns = self.compute_advantages(rewards, values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-12)

        advantages = advantages.detach()
        returns = returns.detach()

        #advantages = torch.where(
        #    advantages > 0,
        #    advantages * 1.5,
        #    advantages
        #)

        dataset = torch.utils.data.TensorDataset(states, action_masks, actions, log_probs_old, returns, advantages)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)

        value_loss, policy_loss, loss = 0, 0, 0
        value = 0

        # torch.autograd.set_detect_anomaly(True)
        self.policy.train()

        for i in range(self.epochs):
            #print(f"{i + 1}/{self.epochs}")
            for j, (mb_states, mb_action_masks, mb_actions, mb_log_probs_old, mb_returns, mb_advantages) in enumerate(
                    loader):
                value, logits = self.policy(mb_states, mb_action_masks)

                dist = torch.distributions.Categorical(logits=logits)
                entropy = dist.entropy().mean()
                log_probs_new = dist.log_prob(mb_actions)

                ratio = torch.exp(log_probs_new - mb_log_probs_old)
                surrogate1 = ratio * mb_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                value_loss = F.mse_loss(value.squeeze(-1), mb_returns)

                loss = policy_loss + self.a1 * value_loss - self.a2 * entropy

                #print(f"Value loss : {value_loss}, Loss : {loss}, Policy loss : {policy_loss}")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        #self.policy.print_special_state(0, STATE_BID_0, MASK_BID_0)
        #self.policy.print_special_state(1, STATE_BID_1, MASK_BID_1)
        #self.policy.print_special_state(2, STATE_BID_2, MASK_BID_2)
        self.writer.add_scalar(tag="Loss/Value", scalar_value=value_loss, global_step=self.train_count)
        self.writer.add_scalar(tag="Loss/Policy", scalar_value=policy_loss, global_step=self.train_count)
        self.writer.add_scalar(tag="Loss/Loss", scalar_value=loss, global_step=self.train_count)
        self.writer.add_scalar(tag="Model/Value", scalar_value=value[0], global_step=self.train_count)
        for name, param in self.policy.named_parameters():
            if 'weight' in name:
                self.writer.add_histogram(f'weights/{name}', param, global_step=self.train_count)
            elif 'bias' in name:
                self.writer.add_histogram(f'biases/{name}', param, global_step=self.train_count)
        self.writer.flush()
        self.train_count += 1
        self.evaluate()

        return value_loss.item(), loss.item()

    def evaluate(self):

        self.policy.eval()

        sim = Simulation(self.policy, [STATE_BID_0, MASK_BID_0, STATE_BID_1, MASK_BID_1, STATE_BID_2, MASK_BID_2])
        winrate, avg_points = sim.start(FIXED_TEST_ROUND)

        #sim_random = Simulation(PPONetwork(), [STATE_BID_0, MASK_BID_0, STATE_BID_1, MASK_BID_1, STATE_BID_2, MASK_BID_2])
        #winrate_random, avg_points_random = sim_random.start(FIXED_TEST_ROUND)

        self.writer.add_scalars("Winrates/Policy",
                                {f"Player {p}": winrate[p] for p in range(self.num_players)},
                                global_step=self.eval_count)
        self.writer.add_scalars("Rewards/Policy",
                                {f"Player {p}": avg_points[p] for p in range(self.num_players)},
                                global_step=self.eval_count)

        #self.writer.add_scalars("Winrates/Random",
        #                        {f"Player {p}": winrate_random[p] for p in range(self.num_players)},
        #                        global_step=self.eval_count)
        #self.writer.add_scalars("Rewards/Random",
        #                        {f"Player {p}": avg_points_random[p] for p in range(self.num_players)},
        #                        global_step=self.eval_count)

        if winrate[self.player_learning] > self.best_winrate:
            self.best_winrate = winrate[self.player_learning]
            self.best_model = copy.deepcopy(self.policy.state_dict())
            if self.DEBUG_PRINT:
                print(f"New best reward : {self.best_winrate}")

            torch.save(self.best_model, os.path.join(self.path, "best_model.pth"))
        torch.save(self.policy.state_dict(), os.path.join(self.path, "last_model.pth"))

        self.policy.train()
        self.eval_count += 1

    @staticmethod
    def parse_batch(batch):

        for k, v in batch.items():
            print(f"{k} : {v}")

    @staticmethod
    def setup_special_state():
        global STATE_BID_0
        global MASK_BID_0
        global STATE_BID_1
        global MASK_BID_1
        global STATE_BID_2
        global MASK_BID_2

        env = Environment()
        env.start_round(2)
        env.bid(0)
        for player in range(1, env.num_players):
            env.bid(0)

        STATE_BID_0 = env.get_state_vector()
        MASK_BID_0 = env.get_action_mask()

        env = Environment()
        env.start_round(2)
        env.bid(1)
        for player in range(1, env.num_players):
            env.bid(0)

        STATE_BID_1 = env.get_state_vector()
        MASK_BID_1 = env.get_action_mask()

        env = Environment()
        env.start_round(2)
        env.bid(2)
        for player in range(1, env.num_players):
            env.bid(0)

        STATE_BID_2 = env.get_state_vector()
        MASK_BID_2 = env.get_action_mask()

        #print(f"{STATE_BID_0=}")
        #print(f"{STATE_BID_1=}")
        #print(f"{STATE_BID_2=}")

    def training_loop(self, iterations):

        value_losses = []
        losses = []

        try:

            self.setup_special_state()

            pbar = trange(iterations)
            for i in pbar:
                batch = self.collect_batch_parallel()
                value_loss, loss = self.ppo_update(batch)
                pbar.set_postfix({
                    "loss": f"{loss:.4f}",
                    "value_loss": f"{value_loss:.4f}"
                })
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
            plt.savefig(os.path.join(self.path, "losses.png"))  # save as PNG
            plt.close()  # optional: closes the figure

        except KeyboardInterrupt:
            print("Training interrupted")
        finally:
            torch.save(self.best_model, os.path.join(self.path, "best_model.pth"))
            torch.save(self.policy.state_dict(), os.path.join(self.path, "last_model.pth"))
            with open(os.path.join(self.path, "value-losses.pkl"), "wb") as f:
                pickle.dump(value_losses, f)
            with open(os.path.join(self.path, "losses.pkl"), "wb") as f:
                pickle.dump(losses, f)
