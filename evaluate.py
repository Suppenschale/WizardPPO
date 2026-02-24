import torch
import torch.multiprocessing as mp
import os
from torch.optim import Adam
from datetime import datetime

from env.bidding_heuristic import bidding_heuristic
from nn.ppo_network import PPONetwork
from sim.simulation import Simulation
from env.environment import Environment
from train.training import Training


def main():
    network = PPONetwork()
    network.load_state_dict(torch.load("log/2025-12-11_01-49-45/model.pth", weights_only=True))
    sim = Simulation(network)
    sim.start()


if __name__ == "__main__":
    main()
