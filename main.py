import numpy as np
import torch
import torch.multiprocessing as mp
from tensorboard import program
from torchviz import make_dot
import os
from torch.optim import Adam
from datetime import datetime

from env.bidding_heuristic import bidding_heuristic
from env.suit import Suit
from nn.card_embedding import CardEmbedding
from nn.ppo_network import PPONetwork
from env.environment import Environment
from sim.simulation import Simulation
from train.training import Training

from env.environment import DECK


def start_tensorboard(path, port=6006):
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", os.path.join(path, "board"), "--port", str(port)])
    url = tb.launch()
    print(f"TensorBoard running at {url}")


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join("log", timestamp)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "board"), exist_ok=True)
    print(f"Create {path}")
    runPath = os.path.join(path, "board")
    print(f"Create {runPath}")

    path_model = os.path.join("log", "2026-02-18_09-13-43")

    state_dict = torch.load(os.path.join(path_model, "last_model.pth"), weights_only=True)

    network = PPONetwork()
    network.load_state_dict(state_dict)

    network.train()
    optimizer = Adam(network.parameters(), lr=1.5e-4)

    start_tensorboard(path)
    train = Training(network, optimizer, path)
    train.training_loop(4000)

    #network = PPONetwork()
    #network.load_state_dict(torch.load("model-2025-07-31_13-03-19.pth", weights_only=True))
    #network = network.to('cpu')
    #sim = Simulation(network)
    #sim.start()


def do_step(env):
    card = env.actions()[0]
    if card.rank == 0:
        card_idx = 52
    elif card.rank == 14:
        card_idx = 53
    else:
        card_idx = (card.suit.value - 1) * 13 + card.rank - 1
    env.step(card_idx)


def simulate(env, to_round, to_trick, to_player):
    env.start_round(3)

    for r in range(1):
        for p in range(env.num_players):
            env.bid(bidding_heuristic(3, 3 / 4))
        for t in range(3):
            for p in range(env.num_players):
                do_step(env)

                if r == to_round - 1 and t == to_trick - 1 and p == to_player - 1:
                    return


def test_special_setup():
    env = Environment()

    env.start_round(2)

    # First player should learn to make 2 tricks
    env.bid(2)
    for p in range(1, env.num_players):
        env.bid(bidding_heuristic(2, 2 / env.num_players))

    for t in range(2):
        for p in range(env.num_players):
            do_step(env)

    print(env.players_points)

def test_network():
    network = PPONetwork()
    env = Environment()

    simulate(env, 5, 1, 1)

    state = env.get_state_vector()
    mask = env.get_action_mask()

    stack = torch.cat([state, state])
    mask_stack = torch.cat([mask, mask])

    out = network(stack, mask_stack)

    print(out)

    dot = make_dot(out)
    dot.render("graph", format="png")


def test_environment():
    env = Environment()

    env.start_game()

    for r in range(env.max_rounds):
        for p in range(env.num_players):
            env.bid(0)
        for t in range(env.num_rounds):
            for p in range(env.num_players):
                card = env.actions()[0]
                if card.rank == 0:
                    card_idx = 52
                elif card.rank == 14:
                    card_idx = 53
                else:
                    card_idx = (card.suit.value - 1) * 13 + card.rank - 1
                env.step(card_idx)

def test_state_vector():
    env = Environment()

    env.start_round(10)

    for p in range(env.num_players):
        env.bid(0)

    print(env.get_state_vector())


if __name__ == "__main__":
    main()
    input("Press any key to end application")
    #test_indexing()
