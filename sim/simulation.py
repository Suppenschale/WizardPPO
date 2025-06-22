import numpy as np
import yaml

from tqdm import tqdm

from env.bidding_heuristic import bidding_heuristic
from env.environment import Environment


class Simulation:

    def __init__(self):
        with open("parameter.yaml", "r") as f:
            config = yaml.safe_load(f)

        self.num_iter = config["sim"]["iter"]
        self.dir = config["sim"]["dir"]
        self.num_players: int = config["env"]["num_players"]

    def start(self):

        stats = np.zeros(self.num_players)
        points = np.empty(self.num_players, dtype=object)

        for i in range(self.num_players):
            points[i] = []

        env = Environment()

        for _ in tqdm(range(self.num_iter), "Simulating"):
            env.reset()
            for r in range(60 // self.num_players):
                env.start_round(r + 1, start_player=r % self.num_players)
                for player in range(self.num_players):
                    env.bid(bidding_heuristic(env.players_hand[player], env.trump))
                for _ in range(r + 1):
                    for _ in range(self.num_players):
                        action = env.actions()
                        card = env.rng.choice(action)
                        env.step(card)

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
