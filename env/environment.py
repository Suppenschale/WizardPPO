import random
from typing import Optional

import yaml
from itertools import product

from env.card import Card

RANKS = [rank for rank in range(0, 15)]
SUITS = ["RED", "YELLOW", "GREEN", "BLUE"]
DECK = [Card(rank, suit) for rank, suit in product(RANKS, SUITS)]


class Environment:

    def __init__(self):

        with open("parameter.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        if self.config["env"]["random"]:
            self.rng = random.Random()
        else:
            self.rng = random.Random(self.config["env"]["seed"])

        self.num_players: int = self.config["env"]["num_players"]
        self.DEBUG_PRINT: bool = self.config["env"]["debug_print"]

        if self.num_players < 2 or 60 % self.num_players != 0:
            raise ValueError(f"Wizard can not be played with {self.num_players} players")

        # Player properties
        self.players_points: list[int] = [0 for _ in range(self.num_players)]
        self.players_hand: list[list[Card]] = [[] for _ in range(self.num_players)]
        self.players_bid: list[int] = [0 for _ in range(self.num_players)]
        self.players_tricks: list[int] = [0 for _ in range(self.num_players)]

        # Card deck
        self.deck: Optional[list[Card]] = DECK.copy()

        # Control flow variables
        self.num_rounds: Optional[int] = None
        self.bidding: Optional[bool] = None
        self.round_counter: int = 0
        self.trick_counter: int = 0
        self.player_counter: int = 0

        # Game logic
        self.cur_player: int = 0
        self.high_player: int = 0
        self.high_card: Optional[Card] = None
        self.trump: Optional[Card] = None
        self.first_card: Optional[Card] = None

    def __str__(self):
        return "Wizard Environment"

    def reset(self) -> None:
        # Player properties
        self.players_points = [0 for _ in range(self.num_players)]
        self.players_hand = [[] for _ in range(self.num_players)]
        self.players_bid = [0 for _ in range(self.num_players)]
        self.players_tricks = [0 for _ in range(self.num_players)]

        # Card deck
        self.deck = DECK.copy()

        # Control flow variables
        self.num_rounds = None
        self.bidding = None
        self.round_counter = 0
        self.trick_counter = 0
        self.player_counter = 0

        # Game logic
        self.cur_player = 0
        self.high_player = 0
        self.high_card = None
        self.trump = None
        self.first_card = None

    def start_round(self, num_rounds: int, start_player: int = 0) -> None:

        # Reset environment
        self.reset()

        # Set start player
        self.cur_player = start_player

        # Begin bidding phase
        self.bidding = True

        # Set number of rounds
        self.num_rounds = num_rounds

        # Shuffle the deck
        self.rng.shuffle(self.deck)

        # Deal the cards
        for _ in range(num_rounds):
            for hand in self.players_hand:
                hand.append(self.deck.pop())

        # Deal the trump
        if self.deck:
            self.trump = self.deck.pop()
        else:
            # placeholder for no card
            self.trump = Card(0, "")

        if self.DEBUG_PRINT:
            print(f"Start of round {num_rounds}")
            print(f"Trump card: {self.trump}")

        # If Wizard is trump chose a random color
        if self.trump.rank == 14:
            self.trump = Card(-1, self.rng.choice(SUITS))
            if self.DEBUG_PRINT:
                print(f"Color chosen : {self.trump.suit}")
                print(f"Trump card: {self.trump}")

        if self.DEBUG_PRINT:
            print("")

    def bid(self, bid: int) -> None:

        # Check if we are in bidding phase
        if self.bidding is None:
            raise ValueError("No round has started yet")

        if not self.bidding:
            raise ValueError("Bidding is only allowed in bidding phase")

        # Check if bid is valid
        if bid < 0 or bid > self.num_rounds:
            raise ValueError(f"Bidding must be range of [0,{self.num_rounds}]")

        # Set players bid
        self.players_bid[self.cur_player] = bid

        # Go to the next player
        self.cur_player = (self.cur_player + 1) % self.num_players

        # Increment player counter
        self.player_counter += 1

        # Bidding phase is over
        if self.player_counter >= self.num_players:
            self.bidding = False
            self.player_counter = 0

    def actions(self) -> list[Card]:

        # If no card was played or first card is Wizard or Jester -> play any card
        if not self.first_card or self.first_card.rank in [0, 14]:
            return self.players_hand[self.cur_player]

        # No suit of first played card -> play any card
        if not {c for c in self.players_hand[self.cur_player]
                if c.suit == self.first_card.suit and c.rank not in [0, 14]}:
            return self.players_hand[self.cur_player]

        return [c for c in self.players_hand[self.cur_player]
                if c.suit == self.first_card.suit or c.rank in [0, 14]]

    def step(self, card: Card) -> None:

        if self.bidding is None:
            raise ValueError("No round has started yet")

        # Check if bidding phase is over
        if self.bidding:
            raise ValueError("Playing is ")

        # Check if step is valid
        if card not in self.players_hand[self.cur_player]:
            raise ValueError("Card is not in players hand")

        # Check if step is legal
        if not self.legal_move(card):
            raise ValueError("Card can not be played")

        if self.DEBUG_PRINT:
            print(f"Player {self.cur_player + 1} hand cards      : {self.players_hand[self.cur_player]}")
            print(f"Player {self.cur_player + 1} allowed actions : {self.actions()}")
            print(f"Player {self.cur_player + 1} plays card      : {card}")
            print("")

        # Remove card from players hand
        self.players_hand[self.cur_player].remove(card)

        # If played card is better than best card, replace high card
        if self.first_card and card.is_higher_than(self.high_card, self.trump):
            self.high_card = card
            self.high_player = self.cur_player

        # If the first played card was a Jester, replace first card
        if self.first_card and self.first_card.rank == "J":
            self.first_card = card

        # Set first card
        if not self.first_card:
            self.first_card = card
            self.high_card = card
            self.high_player = self.cur_player

        # Go to the next player
        self.cur_player = (self.cur_player + 1) % self.num_players

        # Increment player counter
        self.player_counter += 1

        # Trick is over
        if self.player_counter >= self.num_players:
            self.cur_player = self.high_player
            self.players_tricks[self.high_player] += 1
            self.player_counter = 0
            self.round_counter += 1
            self.first_card = None

            if self.DEBUG_PRINT:
                print(f"Trick {self.round_counter} of {self.num_rounds} is over")
                print(f"    Player {self.high_player + 1} has won the trick")
                print("")
                print("Current tricks: ")
                for i in range(self.num_players):
                    print(f"    Player {i + 1}: {self.players_tricks[i]}/{self.players_bid[i]}")
                print("")

        # Playing phase is over
        if self.round_counter >= self.num_rounds:

            self.bidding = None

            # Compute points
            for i in range(self.num_players):
                if self.players_bid[i] == self.players_tricks[i]:
                    self.players_points[i] += 20 + 10 * self.players_bid[i]
                else:
                    self.players_points[i] -= 10 * abs(self.players_bid[i] - self.players_tricks[i])

            if self.DEBUG_PRINT:
                print(f"Round is over")
                print("")
                print("Current points: ")
                for i in range(self.num_players):
                    print(f"    Player {i + 1}: {self.players_points[i]}")
                print("")

    def legal_move(self, card: Card) -> bool:

        # Card must be in players hand
        if card not in self.players_hand[self.cur_player]:
            return False

        # Wizard and Jester can always be played
        if card.rank in [0, 14]:
            return True

        # If the first card played is no Wizard or Jester...
        if self.first_card and self.first_card.rank not in [0, 14]:
            # and played suit is different from first card suit...
            if card.suit != self.first_card.suit:
                # then there must not be a suit in players hand
                if {c for c in self.players_hand[self.cur_player] if c.suit == self.first_card.suit and c.rank not in [0, 14]}:
                    return False

        return True
