import random

import torch
import yaml

from tqdm import tqdm

from env.bidding_heuristic import bidding_heuristic
from env.environment import Environment
from nn.ppo_network import PPONetwork


class Simulation:

    def __init__(self, network: PPONetwork, state_mask: list | None = None):
        with open("parameter.yaml", "r") as f:
            config = yaml.safe_load(f)

        self.iter = config["sim"]["iter"]
        self.dir = config["sim"]["dir"]
        self.num_players: int = config["env"]["num_players"]
        self.game_length = 60 // self.num_players
        self.policy = network

        if state_mask is not None:
            self.STATE_BID_0 = state_mask[0]
            self.MASK_BID_0 = state_mask[1]
            self.STATE_BID_1 = state_mask[2]
            self.MASK_BID_1 = state_mask[3]
            self.STATE_BID_2 = state_mask[4]
            self.MASK_BID_2 = state_mask[5]

    def start(self, num_round):

        stats = [0 for _ in range(self.num_players)]
        points = [[] for _ in range(self.num_players)]
        bids = [[] for _ in range(self.num_players)]
        tricks = [[] for _ in range(self.num_players)]

        for iteration in tqdm(range(self.iter), "Simulating"):
            env = Environment()
            start = random.choice(range(env.num_players))
            env.start_round(num_round, start)

            for p in range(env.num_players):
                if iteration == 0:
                    print()
                    print(f"Player {env.cur_player}: ")
                max_value = -10000000
                max_bid = -1
                for test_bid in range(env.num_rounds + 1):
                    env.players_bid[env.cur_player] = test_bid
                    state = env.get_state_vector()
                    action_mask = env.get_action_mask()
                    value, _ = self.policy(state, action_mask)
                    if iteration == 0:
                        print(f"Expected Future Value for bid {test_bid}: {value.item()}")
                    if value > max_value:
                        max_value = value
                        max_bid = test_bid
                if iteration == 0:
                    print(f"Best bid : {max_bid} ({max_value.item()})")
                    print()
                bids[env.cur_player].append(max_bid)
                env.bid(max_bid)

            for r in range(num_round):

                for i in range(self.num_players):
                    state = env.get_state_vector()
                    action_mask = env.get_action_mask()

                    #if iteration == 0 and r == 0:
                    #    self.policies[i].print_special_state(0, self.STATE_BID_0, self.MASK_BID_0)
                    #    self.policies[i].print_special_state(1, self.STATE_BID_1, self.MASK_BID_1)
                    #    self.policies[i].print_special_state(2, self.STATE_BID_2, self.MASK_BID_2)

                    action, _, _ = self.policy.select_action(state, action_mask)
                    env.step(action)

            for player in range(self.num_players):
                p = env.players_points[player]
                tricks[player].append(env.players_tricks[player])
                points[player].append(p)
                if p == max(env.players_points):
                    stats[player] += 1

        print()
        for p in range(self.num_players):
            print()
            print(f"Player {p + 1} :")
            for i in range(self.iter):
                print(f"  Points = {points[p][i]} ({tricks[p][i]}/{bids[p][i]})")

        winrate = [stats[i] / sum(stats) * 100.0 for i in range(self.num_players)]
        avg = [sum(points[i]) / len(points[i]) for i in range(self.num_players)]

        print("")
        print("Simulation is over")
        print(f"Percentage wins after {self.iter} iterations")
        for i in range(self.num_players):
            print(f"Player {i + 1}: {winrate[i]} (avg. points: {avg[i]})")

        return winrate, avg
