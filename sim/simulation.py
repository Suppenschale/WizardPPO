import random
import yaml

from tqdm import tqdm
from env.environment import Environment
from nn.ppo_network import PPONetwork


class Simulation:

    def __init__(self, network: PPONetwork):
        with open("parameter.yaml", "r") as f:
            config = yaml.safe_load(f)

        self.iter = config["sim"]["iter"]
        self.dir = config["sim"]["dir"]
        self.num_players: int = config["env"]["num_players"]
        //self.DEBUG = config[]
        self.game_length = 60 // self.num_players
        self.policy = network

    def start(self):

        stats = [0 for _ in range(self.num_players)]
        points = [[] for _ in range(self.num_players)]

        for _ in tqdm(range(self.iter), "Simulating"):
            env = Environment()

            for p in range(env.num_players):
                points[p].append(0)

            for num_round in range(1, 16):
                env.start_round(num_round, random.choice(range(env.num_players)))

                for p in range(env.num_players):
                    max_value = float('-inf')
                    max_bid = -1
                    for test_bid in range(env.num_rounds + 1):
                        env.players_bid[env.cur_player] = test_bid
                        state = env.get_state_vector()
                        action_mask = env.get_action_mask()
                        value, _ = self.policy(state, action_mask)
                        if value > max_value:
                            max_value = value
                            max_bid = test_bid
                    env.bid(max_bid)

                for r in range(num_round):

                    for _ in range(self.num_players):
                        state = env.get_state_vector()
                        action_mask = env.get_action_mask()
                        action, _, _ = self.policy.select_action(state, action_mask)
                        env.step(action)

                for player in range(self.num_players):
                    p = env.players_points[player]
                    points[player][-1] += p

            for player in range(self.num_players):
                final_points = [points[player][-1] for player in range(self.num_players)]

                if points[player][-1] == max(final_points):
                    stats[player] += 1

        winrate = [stats[i] / self.iter * 100.0 for i in range(self.num_players)]
        avg = [sum(points[i]) / len(points[i]) for i in range(self.num_players)]

        if self.de
        print("")
        print("Simulation is over")
        print(f"Percentage wins after {self.iter} iterations")
        for i in range(self.num_players):
            print(f"Player {i + 1}: {winrate[i]} (avg. points: {avg[i]})")

        return winrate, avg
