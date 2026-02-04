import os
from tensorboard import program

def start_tensorboard(path, port=6006):
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", os.path.join(path, "board"), "--port", str(port)])
    url = tb.launch()
    print(f"TensorBoard running at {url}")


if __name__ == "__main__":
    path = os.path.join("log", "2026-02-03_22-37-13")
    start_tensorboard(path, port=6007)
    input("Enter key to stop board")