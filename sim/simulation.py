import numpy as np
import torch
import yaml

from tqdm import tqdm

from env.bidding_heuristic import bidding_heuristic
from env.environment import Environment
from nn.ppo_network import PPONetwork


class Simulation:

    def __init__(self, network: PPONetwork):
        with open("parameter.yaml", "r") as f:
            config = yaml.safe_load(f)

        self.iter = config["sim"]["iter"]
        self.dir = config["sim"]["dir"]
        self.num_players: int = config["env"]["num_players"]
        self.game_length = 60 // self.num_players
        self.network = network
        self.network.eval()
        self.policies = [PPONetwork() for _ in range(self.num_players)]

        self.player_learning = 0
        self.policies[self.player_learning] = network

    def start(self):

        stats = np.zeros(self.num_players)
        points = np.empty(self.num_players, dtype=object)

        for i in range(self.num_players):
            points[i] = []


        for _ in tqdm(range(self.iter), "Simulating"):
            env = Environment()
            for T in range(1, self.game_length + 1):
                env.start_round(T)

                for player in range(env.num_players):
        for _ in tqdm(range(self.num_iter), "Simulating"):
            env = Environment()
            for r in range(1, 60 // self.num_players + 1):
                env.start_round(r)

                for player in range(self.num_players):
                    env.bid(bidding_heuristic(env.players_hand[player], env.trump))

                for _ in range(T):

                    start = env.get_start_player()

                    for i in range(self.num_players):
                        state = env.get_state_vector()
                        action_mask = env.get_action_mask()

                        player = (start + i) % self.num_players

                        action, _, _ = self.policies[player].select_action(state, action_mask)
                        env.step(action)

                for _ in range(r):
                    for player in range(self.num_players):
                        state = env.get_state_vector()
                        action_mask = env.get_action_mask()
                        if player == 0 or True:
                            with torch.no_grad():
                                action, _, _ = self.policies[player].select_action(state, action_mask)
                        else:
                            action, _, _ = self.policies[player].select_action(state, action_mask)
                        env.step(action)

            for player in range(self.num_players):
                p = env.players_game_points[player]
                points[player].append(p)
                if p == max(env.players_points):
                    stats[player] += 1
        print("")
        print("Simulation is over")
        print(f"Percentage wins after {self.iter} iterations")
        for i in range(self.num_players):
            print(f"Player {i + 1}: {stats[i] / np.sum(stats) * 100.0} (avg. points: {np.mean(points[i])})")

    def start_round(self, round_number: int):

        stats = np.zeros(self.num_players)
        points = np.empty(self.num_players, dtype=object)

        for i in range(self.num_players):
            points[i] = []

        env = Environment()
        random_poly = PPONetwork()

        for _ in tqdm(range(self.num_iter), "Simulating"):
            env.reset()
            env.start_round(round_number)
            for player in range(self.num_players):
                env.bid(bidding_heuristic(env.players_hand[player], env.trump))
            for _ in range(round_number):
                for player in range(self.num_players):
                    state = env.get_state_vector()
                    action_mask = env.get_action_mask()
                    if player == 0:
                        with torch.no_grad():
                            action, _, _ = self.network.select_action(state, action_mask)
                    else:
                        action, _, _ = random_poly.select_action(state, action_mask)
                    env.step(action)

            for player in range(self.num_players):
                p = env.players_points[player]
                points[player].append(p)
                if p == max(env.players_points):
                    stats[player] += 1

        print("")
        print("Simulation is over")
        print(f"Percentage wins after {self.num_iter} iterations")
        for i in range(self.num_players):
            print(f"Player {i + 1}: {stats[i] / np.sum(stats) * 100.0} (avg. points: {np.mean(points[i])})")
