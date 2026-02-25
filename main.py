import os
import yaml
import warnings

from tensorboard import program
from torch.optim import Adam
from datetime import datetime
from nn.ppo_network import PPONetwork
from train.training import Training

warnings.filterwarnings("ignore", category=UserWarning)


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

    with open("parameter.yaml", "r") as f:
        config = yaml.safe_load(f)

    lr = config["train"]["lr"]
    iterations = config["train"]["iter"]

    network = PPONetwork()
    network.train()
    optimizer = Adam(network.parameters(), lr=lr)

    start_tensorboard(path)
    train = Training(network, optimizer, path)
    train.training_loop(iterations)


if __name__ == "__main__":
    main()
    input("Press any key to end application")
