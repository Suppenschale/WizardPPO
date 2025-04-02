from random import random

import numpy as np
import yaml

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
        env = Environment()

        for i in range(self.num_iter):
            env.reset()
            offset = env.rng.randint(0, 3)
            for r in range(60 // self.num_players):
                env.start_round(r + 1, start_player=r % self.num_players)
                for _ in range(self.num_players):
                    env.bid((r + 1) // self.num_players)
                for _ in range(r + 1):
                    for _ in range(self.num_players):
                        action = env.actions()
                        card = env.rng.choice(action)
                        env.step(card)

            winners = np.array([i for i, p in enumerate(env.players_points) if p == max(env.players_points)])
            stats[winners] += 1

            if i % 1000 == 0:
                print(f"{i}/{self.num_iter}")

        print("")
        print("Simulation is over")
        print(f"Percentage wins after {self.num_iter} iterations")
        for i in range(self.num_players):
            print(f"Player {i + 1}: {stats[i] / np.sum(stats) * 100.0}")


