
import yaml

class Training:

    def __init__(self):
        with open("parameter.yaml", "r") as f:
            config = yaml.safe_load(f)

        self.num_iter = config["train"]["iter"]
        self.dir = config["sim"]["dir"]
        self.num_players: int = config["env"]["num_players"]