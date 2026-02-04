import random

import torch
import yaml

from tqdm import tqdm

from env.bidding_heuristic import bidding_heuristic
from env.environment import Environment
from nn.ppo_network import PPONetwork

class Simulation:

    def __init__(self, networks: list[PPONetwork], state_mask: list):
        with open("parameter.yaml", "r") as f:
            config = yaml.safe_load(f)

        self.iter = config["sim"]["iter"]
        self.dir = config["sim"]["dir"]
        self.num_players: int = config["env"]["num_players"]
        self.game_length = 60 // self.num_players
        self.policies = networks

        self.STATE_BID_0 = state_mask[0]
        self.MASK_BID_0 = state_mask[1]
        self.STATE_BID_1 = state_mask[2]
        self.MASK_BID_1 = state_mask[3]
        self.STATE_BID_2 = state_mask[4]
        self.MASK_BID_2 = state_mask[5]

    def start(self):

        stats = [0 for _ in range(self.num_players)]
        points = [[] for _ in range(self.num_players)]

        for iteration in tqdm(range(self.iter), "Simulating"):
            env = Environment()
            env.start_round(2)

            env.bid(random.choice([2]))
            for player in range(3):
                env.bid(bidding_heuristic(2, 2 / 4))

            for r in range(2):

                start = env.get_start_player()

                for i in range(self.num_players):
                    state = env.get_state_vector()
                    action_mask = env.get_action_mask()

                    player = (start + i) % self.num_players

                    if iteration == 0 and r == 0:
                        self.policies[i].print_special_state(0, self.STATE_BID_0, self.MASK_BID_0)
                        self.policies[i].print_special_state(1, self.STATE_BID_1, self.MASK_BID_1)
                        self.policies[i].print_special_state(2, self.STATE_BID_2, self.MASK_BID_2)

                    action, _, _ = self.policies[player].select_action(state, action_mask)
                    env.step(action)

            for player in range(self.num_players):
                p = env.players_points[player]
                points[player].append(p)
                if p == max(env.players_points):
                    stats[player] += 1
        winrate = [stats[i] / sum(stats) * 100.0 for i in range(self.num_players)]
        avg = [sum(points[i]) / len(points[i]) for i in range(self.num_players)]

        print("")
        print("Simulation is over")
        print(f"Percentage wins after {self.iter} iterations")
        for i in range(self.num_players):
            print(f"Player {i + 1}: {winrate[i]} (avg. points: {avg[i]})")

        return winrate, avg

