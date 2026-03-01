import csv
import os
import random
import tkinter as tk
import argparse
from datetime import datetime

import torch

from env.card import JESTER, WIZARD, Card
from env.environment import Environment, JESTER_POS, SUITS, WIZARD_POS, DUMMY_CARD
from nn.ppo_network import PPONetwork


def action_to_name(action):
    # Jester
    if action == JESTER_POS:
        card = Card(JESTER, SUITS[4])
    # Wizard
    elif action == WIZARD_POS:
        card = Card(WIZARD, SUITS[4])
    # Number card
    else:
        card = Card(action % 13 + 1, SUITS[action // 13])
    return card_to_name(card)


def card_to_name(card):
    suit = card.suit.name.lower()

    if card.rank == -1:
        return f"{suit}"
    if card.rank == JESTER:
        return "jester_1"
    elif card.rank == WIZARD:
        return "wizard_1"
    else:
        return f"{suit}_{card.rank}"


class WizardGameGUI:

    def __init__(self, root, env, network):
        self.root = root
        self.env = env
        self.network = network

        self.bid = None
        self.selected_card = None
        self.next = None
        self.points = [0 for _ in range(env.num_players)]

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = os.path.join("gui", "log")
        os.makedirs(self.path, exist_ok=True)
        self.points_each_round = {f"Player{p}": [] for p in range(env.num_players)}

        root.title("Wizard Game GUI")
        root.geometry("1000x700")

        self.status_label = tk.Label(
            root,
            text="Game started",
            bg="lightgray",
            anchor="w"
        )
        self.status_label.pack(fill=tk.X)

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        self.game_frame = tk.Frame(self.main_frame, bg="darkgreen")
        self.game_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.debug_frame = tk.Frame(self.main_frame, bg="lightgray", width=300)
        self.debug_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.main_frame.update_idletasks()
        total_width = root.winfo_width()
        self.debug_frame.config(width=int(total_width / 3))

        self.build_table_layout()

        self.bid_input_frame = tk.Frame(self.game_frame, bg="lightgray", bd=2, relief=tk.RIDGE)
        self.bid_input_frame.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)  # bottom-right with padding

        tk.Label(self.bid_input_frame, text="Enter your bid:").pack(side=tk.LEFT, padx=5)

        self.bid_entry = tk.Entry(self.bid_input_frame, width=5)
        self.bid_entry.pack(side=tk.LEFT, padx=5)

        self.bid_button = tk.Button(self.bid_input_frame, text="Submit", command=self.submit_bid, state="disabled")
        self.bid_button.pack(side=tk.LEFT, padx=5)

        self.points_frame = tk.Frame(self.game_frame, bg="lightyellow", bd=2, relief=tk.RIDGE)
        self.points_frame.place(relx=1.0, rely=0.0, anchor="ne", x=-10, y=10)  # top-right

        self.points_labels = {}
        for player_id in range(4):
            label = tk.Label(self.points_frame, text=f"Player {player_id}: 0", bg="lightyellow", anchor="w")
            label.pack(fill=tk.X, pady=2)
            self.points_labels[player_id] = label

        self.start_button = tk.Button(
            self.debug_frame,
            text="Start Game",
            command=self.start_game
        )
        self.start_button.pack(pady=10, padx=10)

        self.next_button = tk.Button(
            self.debug_frame,
            text="Next Step",
            command=self.trigger_next_step,
            state="disabled"
        )
        self.next_button.pack(pady=10, padx=10, fill=tk.X)

    def trigger_next_step(self):
        self.next = True

    def start_game(self):
        self.start_button.config(state="disabled")
        self.start_next(1, phase="start")

    def next_step(self, num_round, phase):
        if self.next is None:
            self.next_button.config(state="active")
            self.root.after(100, self.next_step, num_round, phase)
        else:
            self.next = None
            self.next_button.config(state="disabled")
            for player_id in range(self.env.num_players):
                self.player_frames[player_id]["bidding_label"].config(
                    text=f"Bid: {self.env.players_bid[player_id]} | Trick: {self.env.players_tricks[player_id]}")

            if self.env.player_counter == 0:
                for player_id in range(self.env.num_players):
                    self.play_card_middle(player_id, "back")

            if self.env.bidding is None:
                for player_id in range(self.env.num_players):
                    self.points[player_id] += self.env.players_points[player_id]
                    self.points_each_round[f"Player{player_id}"].append(self.points[player_id])
                self.update_player_points()
                self.start_next(num_round + 1, phase="start")
            else:
                self.start_next(num_round, phase="play")

    def start_next(self, num_round, phase):

        num_players = self.env.num_players

        if phase == "start":
            if num_round == 16:
                print("Game is over!")

                with open(os.path.join(self.path, f"{self.timestamp}_scores.csv"), "w", newline="") as f:
                    writer = csv.writer(f)

                    writer.writerow(["Round"] + list(self.points_each_round.keys()))

                    for i in range(15):
                        row = [i + 1] + [self.points_each_round[player][i] for player in self.points_each_round]
                        writer.writerow(row)

                return
            print(f"Start round {num_round}")
            self.env.start_round(num_round, start_player=random.choice(range(num_players)))
            self.setup_round()
            self.start_next(num_round, phase="bid")

        elif phase == "bid":
            print(f"Bid for Player {self.env.cur_player}")
            if self.env.cur_player == 0:
                self.bid_button.config(state="active")
                self.wait_for_player_bid(num_round)
            else:
                max_value = float('-inf')
                max_bid = -1
                for test_bid in range(self.env.num_rounds + 1):
                    self.env.players_bid[self.env.cur_player] = test_bid
                    state = self.env.get_state_vector()
                    action_mask = self.env.get_action_mask()
                    value, _ = self.network(state, action_mask)
                    if value > max_value:
                        max_value = value
                        max_bid = test_bid
                self.set_player_bid(self.env.cur_player, max_bid)
                self.env.bid(max_bid)
                phase = "bid" if self.env.bidding else "play"
                self.start_next(num_round, phase)
        elif phase == "play":
            print(f"Turn for Player {self.env.cur_player}")
            if self.env.cur_player == 0:
                self.enable_hand_cards()
                self.wait_for_player_play(num_round)
            else:
                state = self.env.get_state_vector()
                action_mask = self.env.get_action_mask()
                action, _, _ = self.network.select_action_greedy(state, action_mask)
                card = action_to_name(action)
                self.play_card_middle(self.env.cur_player, card)
                self.env.step(action)
                self.update_hand_card()
                self.next_step(num_round, phase)

    def wait_for_player_bid(self, num_round):
        if self.bid is None:
            self.root.after(100, self.wait_for_player_bid, num_round)
        else:
            self.env.bid(self.bid)
            self.bid_button.config(state="disabled")
            self.bid = None
            phase = "bid" if self.env.bidding else "play"
            self.start_next(num_round, phase)

    def wait_for_player_play(self, num_round):
        if self.selected_card is None:
            self.root.after(100, self.wait_for_player_play, num_round)
        else:
            card = self.selected_card
            self.selected_card = None
            suit, rank = card.split("_")
            suit_to_index = {s: i for i, s in enumerate(['red', 'yellow', 'green', 'blue'])}
            if suit == "jester":
                action = JESTER_POS
            elif suit == "wizard":
                action = WIZARD_POS
            else:
                action = (suit_to_index[suit]) * 13 + int(rank) - 1
            self.play_card_middle(self.env.cur_player, card)
            self.env.step(action)
            self.disable_hand_cards()
            self.update_hand_card()
            for player_id in range(self.env.num_players):
                self.player_frames[player_id]["bidding_label"].config(
                    text=f"Bid: {self.env.players_bid[player_id]} | Trick: {self.env.players_tricks[player_id]}")

            if self.env.player_counter == 0:
                for player_id in range(self.env.num_players):
                    self.play_card_middle(player_id, "back")

            if self.env.bidding is None:
                for player_id in range(self.env.num_players):
                    self.points[player_id] += self.env.players_points[player_id]
                    self.points_each_round[f"Player{player_id}"].append(self.points[player_id])
                self.update_player_points()
                self.start_next(num_round + 1, phase="start")
            else:
                self.start_next(num_round, phase="play")

    def update_hand_card(self):
        self.show_player_cards(0, [card_to_name(card) for card in self.env.players_hand[0]])
        self.show_player_cards(1, ["back" for _ in range(len(self.env.players_hand[1]))])
        self.show_player_cards(2, ["back" for _ in range(len(self.env.players_hand[2]))])
        self.show_player_cards(3, ["back" for _ in range(len(self.env.players_hand[3]))])

    def setup_round(self):

        trump = self.env.trump

        if trump == DUMMY_CARD or trump.rank == JESTER:
            color = "white"
        else:
            color = trump.suit.name.lower()

        self.set_trump_card(card_to_name(trump), color)
        self.update_hand_card()

        self.bid = None
        self.selected_card = None

        self.update_player_points()

        self.bid_button.config(state="disabled")
        self.disable_hand_cards()

        for player_id in range(self.env.num_players):
            self.player_frames[player_id]["bidding_label"].config(text="Bid: - | Trick: -")
            self.play_card_middle(player_id, "back")

    def build_table_layout(self):
        self.game_frame.grid_rowconfigure(0, weight=1)
        self.game_frame.grid_rowconfigure(1, weight=1)
        self.game_frame.grid_rowconfigure(2, weight=1)

        self.game_frame.grid_columnconfigure(0, weight=1)
        self.game_frame.grid_columnconfigure(1, weight=1)
        self.game_frame.grid_columnconfigure(2, weight=1)

        self.player_frames = {}

        self.create_player_area(0, 2, 1)
        self.create_player_area(1, 1, 0)
        self.create_player_area(2, 0, 1)
        self.create_player_area(3, 1, 2)

        self.card_image_cache = {}

        self.card_widgets = {}

        self.middle_frame = tk.Frame(self.game_frame, bg="darkgreen")
        self.middle_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)

        self.middle_frame.grid_rowconfigure(0, weight=1)
        self.middle_frame.grid_rowconfigure(1, weight=1)
        self.middle_frame.grid_columnconfigure(0, weight=1)
        self.middle_frame.grid_columnconfigure(1, weight=1)

        self.trick_frame = tk.Frame(self.middle_frame, bg="darkgreen")
        self.trick_frame.grid(row=1, column=1, sticky="nsew")

        self.trump_label = tk.Label(self.middle_frame, text="Trump", bg="black")
        self.trump_label.grid(row=0, column=1, sticky="ne", padx=5, pady=5)

        self.played_card_labels = {}

    def update_player_points(self):
        for player_id in range(self.env.num_players):
            self.points_labels[player_id].config(text=f"Player {player_id}: {self.points[player_id]}")

    def submit_bid(self):
        bid_text = self.bid_entry.get()

        try:
            bid = int(bid_text)
            self.set_player_bid(0, bid)
            self.bid_entry.delete(0, tk.END)
            self.bid = bid

        except ValueError:
            self.status_label.config(text="Invalid bid! Enter a number.")

    def show_player_cards(self, player_id, cards):

        cards_frame = self.player_frames[player_id]["cards_frame"]

        for widget in cards_frame.winfo_children():
            widget.destroy()

        self.card_widgets[player_id] = []

        cards = cards[:15]

        if player_id in [1, 3]:
            side = tk.TOP
            padx, pady = 2, 2
        else:
            side = tk.LEFT
            padx, pady = 2, 2

        for i, card in enumerate(cards):

            if player_id == 0:
                img = self.load_card_image(f"{card}_{player_id}", size=2)
            else:
                img = self.load_card_image(f"{card}_{player_id}")

            if player_id == 0:
                label = tk.Label(
                    cards_frame,
                    image=img,
                    cursor="hand2",
                    bd=2,
                    relief=tk.RAISED
                )
                label.image = img

            else:
                label = tk.Label(
                    cards_frame,
                    image=img,
                    bd=1,
                    relief=tk.FLAT
                )
                label.image = img

            label.pack(side=side, padx=padx, pady=pady)
            label.card = card
            label.i = i

            self.card_widgets[player_id].append(label)

    def load_card_image(self, card_name, size=3):

        path = os.path.join("gui", "cards", f"{card_name}.png")

        try:
            img = tk.PhotoImage(file=path)
            img = img.subsample(size, size)

        except Exception:
            img = tk.PhotoImage(width=60, height=90)

        self.card_image_cache[card_name] = img

        return img

    def create_player_area(self, player_id, row, col):
        frame = tk.Frame(
            self.game_frame,
            bd=2,
            relief=tk.GROOVE,
            bg="white"
        )

        frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

        label = tk.Label(frame, text=f"Player {player_id}")
        label.pack(side=tk.TOP)

        bidding_label = tk.Label(frame, text="Bid: - | Trick: -")
        bidding_label.pack(side=tk.TOP, pady=(0, 5))

        cards_frame = tk.Frame(frame)
        cards_frame.pack(expand=True)

        self.player_frames[player_id] = {
            "frame": frame,
            "cards_frame": cards_frame,
            "bidding_label": bidding_label
        }

    def set_player_bid(self, player_id, bid):
        if player_id in self.player_frames:
            label = self.player_frames[player_id]["bidding_label"]
            label.config(text=f"Bid: {bid} | Trick: 0")

    def disable_hand_cards(self):
        for label in self.card_widgets[0]:
            label.unbind("<Button-1>")

    def enable_hand_cards(self):

        actions = [card_to_name(card) for card in self.env.actions()]
        for label in self.card_widgets[0]:
            if label.card in actions:
                label.bind(
                    "<Button-1>",
                    lambda e, c=label.card, idx=label.i: self.on_card_clicked(c, idx)
                )

    def on_card_clicked(self, card, index):

        self.status_label.config(
            text=f"You clicked {card}"
        )

        self.selected_card = card

    def play_card_middle(self, player_id, card_name):
        img = self.load_card_image(f"{card_name}_{player_id}", size=1)

        if player_id in self.played_card_labels:
            self.played_card_labels[player_id].destroy()

        label = tk.Label(self.trick_frame, image=img, bd=2, relief=tk.RAISED)
        label.image = img

        if player_id == 0:
            label.grid(row=2, column=1, pady=5)
        elif player_id == 1:
            label.grid(row=1, column=0, padx=5)
        elif player_id == 2:
            label.grid(row=0, column=1, pady=5)
        elif player_id == 3:
            label.grid(row=1, column=2, padx=5)

        self.played_card_labels[player_id] = label

    def set_trump_card(self, card_name, color):
        self.trump_label.config(bg=color)
        img = self.load_card_image(f"{card_name}_0", size=1)
        self.trump_label.config(image=img, text="")
        self.trump_label.image = img


def main():
    print("Start Game!")

    parser = argparse.ArgumentParser(description="Run Wizard PPO GUI")
    parser.add_argument(
        "--model-path",
        required=False,
        default=None,
        help="Path to the trained model file"
    )

    args = parser.parse_args()

    env = Environment(path="parameter.yaml")
    network = PPONetwork(path="parameter.yaml")

    if args.model_path is not None and not os.path.exists(args.model_path):
        raise FileNotFoundError(f"The path '{args.model_path}' does not exist.")

    if args.model_path is not None:
        model_dict = torch.load(args.model_path, weights_only=True)
        network.load_state_dict(model_dict)

    root = tk.Tk()

    WizardGameGUI(root, env, network)
    root.mainloop()


if __name__ == "__main__":
    main()
