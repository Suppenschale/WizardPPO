import os
import random
import pickle

import torch
from tqdm import tqdm

from env.environment import Environment
from nn.ppo_network import PPONetwork


def start(iter, policies, path):
    env = Environment()
    stats = [0 for _ in range(env.num_players)]
    points = [[] for _ in range(env.num_players)]
    correct_bidding = {(r + 1): 0 for r in range(15)}
    points_per_round = {(r + 1): {p: [] for p in range(env.num_players)} for r in range(15)}

    for iteration in tqdm(range(iter), "Simulating"):
        env = Environment()

        for p in range(env.num_players):
            points[p].append(0)

        for num_round in range(1, 16):
            env.start_round(num_round, random.choice(range(env.num_players)))

            for p in range(env.num_players):
                max_value = -10000000
                max_bid = -1
                for test_bid in range(env.num_rounds + 1):
                    env.players_bid[env.cur_player] = test_bid
                    state = env.get_state_vector()
                    action_mask = env.get_action_mask()
                    value, _ = policies[env.cur_player](state, action_mask)
                    if value > max_value:
                        max_value = value
                        max_bid = test_bid
                env.bid(max_bid)

            for r in range(num_round):

                for i in range(env.num_players):
                    state = env.get_state_vector()
                    action_mask = env.get_action_mask()

                    # if iteration == 0 and r == 0:
                    #    self.policies[i].print_special_state(0, self.STATE_BID_0, self.MASK_BID_0)
                    #    self.policies[i].print_special_state(1, self.STATE_BID_1, self.MASK_BID_1)
                    #    self.policies[i].print_special_state(2, self.STATE_BID_2, self.MASK_BID_2)

                    action, _, _ = policies[env.cur_player].select_action_greedy(state, action_mask)
                    env.step(action)

            for player in range(env.num_players):
                p = env.players_points[player]
                points[player][-1] += p

                points_per_round[num_round][player].append(p)
                if p > 0:
                    correct_bidding[num_round] += 1 / (env.num_players * iter)

        for player in range(env.num_players):
            final_points = [points[player][-1] for player in range(env.num_players)]

            if points[player][-1] == max(final_points):
                stats[player] += 1

    winrate = [stats[i] / iter * 100.0 for i in range(env.num_players)]
    avg = [sum(points[i]) / len(points[i]) for i in range(env.num_players)]

    save = {
        "correct_bidding": correct_bidding,
        "points_per_round": points_per_round,
        "winrate": winrate,
        "avg": avg,
        "num_players": env.num_players,
        "iter": iter
    }

    with open(path, "wb") as f:
        pickle.dump(save, f)


def simulate(path_to_model):
    incomplete_policy = PPONetwork()
    final_policy = PPONetwork()
    random_policy = PPONetwork()

    incomplete_dict = torch.load(os.path.join(path_to_model, "last_model.pth"), weights_only=True)
    final_dict = torch.load(os.path.join(path_to_model, "extend1", "last_model.pth"), weights_only=True)

    incomplete_policy.load_state_dict(incomplete_dict)
    final_policy.load_state_dict(final_dict)

    iterations = 1000

    start(iterations, [random_policy, random_policy, random_policy, random_policy], os.path.join("final_0_random_4.pkl"))
    start(iterations, [final_policy, random_policy, random_policy, random_policy], os.path.join("final_1_random_3.pkl"))
    start(iterations, [final_policy, final_policy, random_policy, random_policy], os.path.join("final_2_random_2.pkl"))
    start(iterations, [final_policy, final_policy, final_policy, random_policy], os.path.join("final_3_random_1.pkl"))
    start(iterations, [final_policy, final_policy, final_policy, final_policy], os.path.join("final_4_random_0.pkl"))
    start(iterations, [final_policy, incomplete_policy, final_policy, incomplete_policy], os.path.join("final_2_sub_2.pkl"))


def main():
    path = os.path.join("save", "final")
    simulate(path)


if __name__ == "__main__":
    main()
