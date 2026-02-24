import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def read_files_actions(path):
    action_red_files = [f"run-Action_Bid {p}_Red 10-tag-Action_Bid {p}.csv" for p in range(3)]
    action_yellow_files = [f"run-Action_Bid {p}_Yellow 13-tag-Action_Bid {p}.csv" for p in range(3)]

    action_red = {f"Bid {p}": pd.read_csv(os.path.join(path, action_red_files[p])) for p in range(3)}
    action_yellow = {f"Bid {p}": pd.read_csv(os.path.join(path, action_yellow_files[p])) for p in range(3)}

    return action_red, action_yellow


def read_files_reward_winrate(path):
    reward_files = [f"run-Rewards_Policy_Player {p}-tag-Rewards_Policy.csv" for p in range(4)]
    winrate_files = [f"run-Winrates_Policy_Player {p}-tag-Winrates_Policy.csv" for p in range(4)]

    rewards = {f"Player {p}": pd.read_csv(os.path.join(path, reward_files[p])) for p in range(4)}
    winrates = {f"Player {p}": pd.read_csv(os.path.join(path, winrate_files[p])) for p in range(4)}

    return rewards, winrates


def read_file_simulation(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    winrate = data["winrate"]
    avg = data["avg"]
    correct_bidding = data["correct_bidding"]
    points_per_round = data["points_per_round"]
    num_players = data["num_players"]
    iterations = data["iter"]

    return winrate, avg, correct_bidding, points_per_round, num_players, iterations


def read_files_reward_winrate_final(paths):
    rewards = {f"Player {p}": pd.DataFrame() for p in range(4)}
    winrates = {f"Player {p}": pd.DataFrame() for p in range(4)}

    for path in paths:
        reward_files = [f"run-Rewards_Policy_Player {p}-tag-Rewards_Policy.csv" for p in range(4)]
        winrate_files = [f"run-Winrates_Policy_Player {p}-tag-Winrates_Policy.csv" for p in range(4)]

        for p in range(4):
            df_reward_total = rewards[f"Player {p}"]
            df_winrate_total = winrates[f"Player {p}"]

            start = len(df_reward_total)

            df_reward = pd.read_csv(os.path.join(path, reward_files[p]))
            df_reward["Step"] = range(start, start + len(df_reward))

            df_winrate = pd.read_csv(os.path.join(path, winrate_files[p]))
            df_winrate["Step"] = range(start, start + len(df_winrate))

            rewards[f"Player {p}"] = pd.concat([df_reward_total, df_reward], ignore_index=True)
            winrates[f"Player {p}"] = pd.concat([df_winrate_total, df_winrate], ignore_index=True)

    return rewards, winrates


def plot_rewards_winrates_round2(rewards, winrates):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    colors = ['tab:red', 'tab:purple', 'tab:green', 'tab:blue']

    axs[0].axhline(y=335 / 48, linestyle='--', color="black")

    for key, color in zip(rewards, colors):
        smoothed = gaussian_filter1d(rewards[key]["Value"], sigma=15)
        axs[0].plot(rewards[key]["Step"], smoothed, label=key, color=color)
        axs[0].plot(rewards[key]["Step"], rewards[key]["Value"], color=color, alpha=0.3)
        axs[0].set_title(f"Rewards")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Reward")
        axs[0].grid(True)
        axs[0].legend()

    for key, color in zip(winrates, colors):
        smoothed = gaussian_filter1d(winrates[key]["Value"], sigma=15)
        axs[1].plot(winrates[key]["Step"], smoothed, label=key, color=color)
        axs[1].plot(winrates[key]["Step"], winrates[key]["Value"], color=color, alpha=0.3)
        axs[1].set_title(f"Win rates")
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Win rate [%]")
        axs[1].grid(True)
        axs[1].set_ylim(0, 100)
        axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join("..", "BachelorThesis", "content", "pics", "reward_winrate_round2.pdf"))
    #plt.show()


def plot_rewards_winrates_round11(rewards, winrates):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    colors = ['tab:red', 'tab:purple', 'tab:green', 'tab:blue']

    for key, color in zip(rewards, colors):
        smoothed = gaussian_filter1d(rewards[key]["Value"], sigma=25)
        axs[0].plot(rewards[key]["Step"], smoothed, label=key, color=color)
        axs[0].plot(rewards[key]["Step"], rewards[key]["Value"], color=color, alpha=0.3)
        axs[0].set_title(f"Rewards")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Reward")
        axs[0].grid(True)
        axs[0].legend()

    for key, color in zip(winrates, colors):
        smoothed = gaussian_filter1d(winrates[key]["Value"], sigma=15)
        axs[1].plot(winrates[key]["Step"], smoothed, label=key, color=color)
        axs[1].plot(winrates[key]["Step"], winrates[key]["Value"], color=color, alpha=0.3)
        axs[1].set_title(f"Win rates")
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Win rate [%]")
        axs[1].grid(True)
        axs[1].set_ylim(0, 100)
        axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join("..", "BachelorThesis", "content", "pics", "reward_winrate_round11.pdf"))
    #plt.show()


def plot_rewards_winrates_final(rewards, winrates):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    colors = ['tab:red', 'tab:purple', 'tab:green', 'tab:blue']

    for key in rewards:
        rewards[key] = rewards[key].iloc[:1000]
        winrates[key] = winrates[key].iloc[:1000]

    for key, color in zip(rewards, colors):
        smoothed = gaussian_filter1d(rewards[key]["Value"], sigma=100)
        axs[0].plot(rewards[key]["Step"], smoothed, label=key, color=color)
        axs[0].plot(rewards[key]["Step"], rewards[key]["Value"], color=color, alpha=0.3)
        axs[0].set_title(f"Rewards")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Reward")
        axs[0].grid(True)
        axs[0].legend()

    for key, color in zip(winrates, colors):
        smoothed = gaussian_filter1d(winrates[key]["Value"], sigma=50)
        axs[1].plot(winrates[key]["Step"], smoothed, label=key, color=color)
        axs[1].plot(winrates[key]["Step"], winrates[key]["Value"], color=color, alpha=0.3)
        axs[1].set_title(f"Win rates")
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Win rate [%]")
        axs[1].grid(True)
        axs[1].set_ylim(0, 100)
        axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join("..", "BachelorThesis", "content", "pics", "reward_winrate_final.pdf"))
    #plt.show()


def plot_actions(action_red, action_yellow):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    for i, ax in enumerate(axs):
        ax.plot(action_red[f"Bid {i}"]["Step"], action_red[f"Bid {i}"]["Value"], label="Red 10", color='r')
        ax.plot(action_yellow[f"Bid {i}"]["Step"], action_yellow[f"Bid {i}"]["Value"], label="Yellow 13", color='y')
        ax.grid(True)
        ax.legend()
        ax.set_title(f"Policy for {i} Bid")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Probability of choosing action")

    plt.tight_layout()
    plt.savefig(os.path.join("..", "BachelorThesis", "content", "pics", "policy_round2.pdf"))
    #plt.show()


def plot_graphs_round2(rewards, winrates, action_red, action_yellow):
    plot_rewards_winrates_round2(rewards, winrates)
    plot_actions(action_red, action_yellow)


def plot_graphs_round11(rewards, winrates):
    plot_rewards_winrates_round11(rewards, winrates)


def plot_graphs_final(rewards, winrates):
    plot_rewards_winrates_final(rewards, winrates)


def plot_correct_bids(correct_bidding, correct_bidding_random):

    plt.figure(figsize=(10, 5))

    rounds = np.array(list(correct_bidding.keys()))
    percentage = [correct_bidding[key] * 100 for key in correct_bidding]
    percentage_random = [correct_bidding_random[key] * 100 for key in correct_bidding_random]
    labels = [f"Round {r}" for r in correct_bidding]
    width = 0.35

    plt.bar(rounds - width / 2, percentage, width, label="Learned", zorder=3)
    plt.bar(rounds + width / 2, percentage_random, width, label="Random", zorder=3)

    plt.title("Percentage of Correctly Bid Tricks per Round")
    plt.ylabel("Bidding Accuracy [%]")
    plt.xticks(rounds, labels, rotation=45)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join("..", "BachelorThesis", "content", "pics", "correct_bids.pdf"))
    # plt.show()


def print_stats(winrate, avg, filename):

    data = {
        f"Player{p}": [winrate[p], avg[p]] for p in range(4)
    }
    data["Name"] = ["Winrate [%]", "Reward"]
    df = pd.DataFrame(data)
    df.to_csv(os.path.join("..", "BachelorThesis", "content", "table", f"{filename}.csv"), index=False, float_format="%.2f")


def plot_cumulative_points(points_per_round, randoms, title):

    plt.figure(figsize=(10, 5))

    values = [[] for _ in range(4)]

    for key in points_per_round:
        for p in points_per_round[key]:
            values[p].append(np.array(points_per_round[key][p]).mean())

    rounds = list(points_per_round.keys())
    labels = [f"Round {r + 1}" for r in range(15)]

    for p in range(4):
        if p < 4 - randoms:
            label = "Learned"
        else:
            label = "Random"
        plt.step(rounds, np.cumsum(values[p]), label=label)

    plt.title(title)
    plt.xticks(rounds, labels, rotation=45)
    plt.ylabel("Cumulative Reward")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("..", "BachelorThesis", "content", "pics", "cumulative_points.pdf"))
    # plt.show()


def plot_simulations(path):
    files = [os.path.join(path, f"final_{r}_random_{4 - r}.pkl") for r in range(5)]
    files.append(os.path.join(path, f"final_2_sub_2.pkl"))

    data = {f"final_{r}_random_{4 - r}": read_file_simulation(files[r]) for r in range(5)}
    data["final_2_sub_2"] = read_file_simulation(files[-1])

    # Plot 4 final vs 0 random
    plot_correct_bids(data["final_4_random_0"][2], data["final_0_random_4"][2])

    # Plot 1 final vs 3 random
    plot_cumulative_points(data["final_1_random_3"][3], 3, "Cumulative Reward per Player")

    for d in data:
        print_stats(data[d][0], data[d][1], d)


def main():
    #path = os.path.join("save", "round_2")
    #rewards, winrates = read_files_reward_winrate(path)
    #red, yellow = read_files_actions(path)
    #plot_graphs_round2(rewards, winrates, red, yellow)

    #path = os.path.join("save", "round_11")
    #rewards, winrates = read_files_reward_winrate(path)
    #plot_graphs_round11(rewards, winrates)

    #paths = [os.path.join("save", "final"), os.path.join("save", "final", "extend1")]
    #rewards, winrates = read_files_reward_winrate_final(paths)
    #plot_graphs_final(rewards, winrates)

    path = os.path.join(".")
    plot_simulations(path)


if __name__ == "__main__":
    main()
