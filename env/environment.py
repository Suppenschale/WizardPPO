import random
import torch
from typing import Optional
import torch.nn.functional as F

import yaml
from itertools import product

from env.card import Card
from env.card import JESTER, WIZARD
from env.suit import Suit

RANKS = [rank for rank in range(0, 15)]
SUITS = [suit for suit in Suit]
DECK = [Card(rank, suit) for rank, suit in product(RANKS, SUITS) if suit != Suit.NO_SUIT]
DUMMY_CARD = Card(-1, Suit.NO_SUIT)

JESTER_POS = 52
WIZARD_POS = 53


def one_hot_encode_cards(cards: list[Card]) -> list:
    one_hot = [0 for _ in range(54)]
    suit_to_index = {s: i for i, s in enumerate(SUITS)}

    for card in cards:
        if card.rank == JESTER:
            one_hot[JESTER_POS] = 1
        elif card.rank == WIZARD:
            one_hot[WIZARD_POS] = 1
        else:
            idx = (suit_to_index[card.suit]) * 13 + card.rank - 1
            one_hot[idx] = 1
    return one_hot


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
        self.DEBUG_ROUND: bool = self.config["env"]["debug_round"]
        self.ROUND: int = self.config["env"]["round"]

        self.emb_dim = self.config["embedding"]["emb_dim"]
        self.deck_size = self.config["env"]["deck_size"]

        if self.num_players < 2 or 60 % self.num_players != 0:
            raise ValueError(f"Wizard can not be played with {self.num_players} players")
        self.max_rounds = 60 // self.num_players

        # Player properties
        self.players_game_points: list[int] = [0 for _ in range(self.num_players)]
        self.players_points: list[int] = [0 for _ in range(self.num_players)]
        self.players_hand: list[list[Card]] = [[] for _ in range(self.num_players)]
        self.players_bid: list[int] = [0 for _ in range(self.num_players)]
        self.players_tricks: list[int] = [0 for _ in range(self.num_players)]

        # Card deck
        self.deck: Optional[list[Card]] = DECK.copy()

        # Control flow variables
        self.game_is_running: bool = False
        self.num_rounds: int = 0
        self.bidding: Optional[bool] = None
        self.round_counter: int = 0
        self.trick_counter: int = 0
        self.player_counter: int = 0

        # Game logic
        self.start_player: int = 0
        self.cur_player: int = 0
        self.high_player: int = 0
        self.high_card: Optional[Card] = None
        self.trump: Card = DUMMY_CARD
        self.first_card: Optional[Card] = None

        self.cards_played_in_trick: list[Card] = []
        self.cards_played: list[Card] = []
        self.players_round_points_history: list[list[int]] = [[] for _ in range(self.max_rounds)]

    def __str__(self):
        return "Wizard Environment"

    def reset(self) -> None:

        self.players_game_points = [0 for _ in range(self.num_players)]
        self.players_round_points_history: list[list[int]] = [[] for _ in range(self.max_rounds)]

        self.num_rounds = 0

        self.start_player = 0

        self.reset_round()

    def reset_round(self) -> None:
        # Player properties
        self.players_points = [0 for _ in range(self.num_players)]
        self.players_hand = [[] for _ in range(self.num_players)]
        self.players_bid = [0 for _ in range(self.num_players)]
        self.players_tricks = [0 for _ in range(self.num_players)]

        # Card deck
        self.deck = DECK.copy()

        # Control flow variables
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

        # Played cards
        self.cards_played_in_trick = []
        self.cards_played = []

    def start_game(self, start_player: int = 0) -> None:
        # Reset env
        self.reset()

        # Start game = true
        self.game_is_running = True

        # Set start player
        self.start_player = start_player

        # Start first round
        self.start_round()

    def start_round(self) -> None:

        # Reset round
        self.reset_round()

        # Set start player
        self.cur_player = self.start_player

        # Set next start player for next round
        self.start_player = (self.start_player + 1) % self.num_players

        # Begin bidding phase
        self.bidding = True

        # Increase num_rounds
        self.num_rounds = self.num_rounds + 1

        # Shuffle the deck
        self.rng.shuffle(self.deck)

        # Deal the cards
        for _ in range(self.num_rounds):
            for hand in self.players_hand:
                hand.append(self.deck.pop())

        # Deal the trump
        if self.deck:
            self.trump = self.deck.pop()
            self.cards_played.append(self.trump)
        else:
            # Last round, set dummy trump
            self.trump = DUMMY_CARD

        if self.DEBUG_PRINT or (self.DEBUG_ROUND and self.num_rounds == self.ROUND):
            print(f"Start of round {self.num_rounds}")
            print(f"Trump card: {self.trump}")

        # If Wizard is trump chose a random color
        if self.trump.rank == WIZARD:
            self.trump = Card(-1, self.rng.choice(SUITS))
        if self.DEBUG_PRINT or (self.DEBUG_ROUND and self.num_rounds == self.ROUND):
                print(f"Color chosen : {self.trump.suit}")
                print(f"Trump card: {self.trump}")

        if self.DEBUG_PRINT or (self.DEBUG_ROUND and self.num_rounds == self.ROUND):
            print("")

    def bid(self, bid: int) -> None:

        # Check if round has stared
        if self.bidding is None:
            raise ValueError("No round has started yet")

        # Check if we are in bidding phase
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
        if not self.first_card or self.first_card.rank in [JESTER, WIZARD]:
            return self.players_hand[self.cur_player]

        # No suit of first played card -> play any card
        if not {c for c in self.players_hand[self.cur_player]
                if c.suit == self.first_card.suit and c.rank not in [JESTER, WIZARD]}:
            return self.players_hand[self.cur_player]

        return [c for c in self.players_hand[self.cur_player]
                if c.suit == self.first_card.suit or c.rank in [JESTER, WIZARD]]

    def step(self, card_idx: int) -> None:

        # Check if game has started
        if not self.game_is_running:
            raise ValueError("No game has started yet")

        # Check if (any) round has started
        if self.bidding is None:
            raise ValueError("No round has stared yet")

        # Check if bidding phase is over
        if self.bidding:
            raise ValueError("Bidding is not over yet")

        # Compute card
        # Jester
        if card_idx == JESTER_POS:
            card = Card(JESTER, SUITS[4])
        # Wizard
        elif card_idx == WIZARD_POS:
            card = Card(WIZARD, SUITS[4])
        # Number card
        else:
            card = Card(card_idx % 13 + 1, SUITS[card_idx // 13])

        # Check if step is valid
        if card not in self.players_hand[self.cur_player]:
            raise ValueError("Card is not in players hand")

        # Check if step is legal
        if not self.legal_move(card):
            raise ValueError("Card can not be played")

        if self.DEBUG_PRINT or (self.DEBUG_ROUND and self.num_rounds == self.ROUND):
            print(f"Player {self.cur_player + 1} hand cards      : {self.players_hand[self.cur_player]}")
            print(f"Player {self.cur_player + 1} allowed actions : {self.actions()}")
            print(f"Player {self.cur_player + 1} plays card      : {card}")
            print("")

        # Remove card from players hand
        self.players_hand[self.cur_player].remove(card)

        # Add played card to history
        self.cards_played_in_trick.append(card)
        self.cards_played.append(card)

        # If played card is better than best card, replace high card
        if self.first_card and card.is_higher_than(self.high_card, self.trump):
            self.high_card = card
            self.high_player = self.cur_player

        # If the first played card was a Jester, replace first card
        if self.first_card and self.first_card.rank == JESTER:
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
            self.cards_played_in_trick = []

            if self.DEBUG_PRINT or (self.DEBUG_ROUND and self.num_rounds == self.ROUND):
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
                    points = 20 + 10 * self.players_bid[i]
                else:
                    points = -10 * abs(self.players_bid[i] - self.players_tricks[i])
                self.players_game_points[i] += points
                self.players_round_points_history[self.num_rounds - 1].append(points)
                self.players_points[i] = points

            if self.DEBUG_PRINT or (self.DEBUG_ROUND and self.num_rounds == self.ROUND):
                print(f"Round is over")
                print("")
                print("Round points: ")
                for i in range(self.num_players):
                    print(f"    Player {i + 1}: {self.players_points[i]}")
                print("")

            # Start next round
            if self.num_rounds < self.max_rounds:
                self.start_round()
            else:
                self.game_is_running = False

                if self.DEBUG_PRINT or (self.DEBUG_ROUND and self.num_rounds == self.ROUND):
                    print(f"Game is over")
                    print("")
                    print("Round histories:")
                    for r in range(self.max_rounds):
                        print(f"    Round {r}:")
                        for i in range(self.num_players):
                            print(f"        Player {i + 1}: {self.players_round_points_history[r][i]}")
                    print("Game points: ")
                    for i in range(self.num_players):
                        print(f"    Player {i + 1}: {self.players_game_points[i]}")
                    print("")

    def legal_move(self, card: Card) -> bool:

        # Card must be in players hand
        if card not in self.players_hand[self.cur_player]:
            return False

        # Wizard and Jester can always be played
        if card.rank in [JESTER, WIZARD]:
            return True

        # If the first card played is no Wizard or Jester...
        if self.first_card and self.first_card.rank not in [JESTER, WIZARD]:
            # and played suit is different from first card suit...
            if card.suit != self.first_card.suit:
                # then there must not be a suit in players hand
                if {c for c in self.players_hand[self.cur_player] if c.suit == self.first_card.suit and c.rank not in [JESTER, WIZARD]}:
                    return False

        return True

    def get_start_player(self):
        return self.cur_player

    def beat_current_high_card(self) -> set[Card]:

        # If no card is played, all cards can win the trick
        if self.high_card is None:
            return set(DECK)

        # Collect all cards which can beat high card
        cards_higher_than_high_card = set()
        for card in DECK:
            if card.is_higher_than(self.high_card, self.trump):
                cards_higher_than_high_card.add(card)

        # Trump can not beat high card since it can not be played
        if self.trump:
            cards_higher_than_high_card.discard(self.trump)

        return cards_higher_than_high_card.difference(self.cards_played).difference(self.players_hand[self.cur_player])

    def get_action_mask(self) -> torch.Tensor:
        # add batch dim (1, 54)
        return torch.tensor(one_hot_encode_cards(self.actions())).type(torch.long).unsqueeze(0)

    def get_state_vector(self) -> torch.tensor:
        hand = self.players_hand[self.cur_player]
        card_left = len(self.players_hand[self.cur_player])
        num_of_wizards = sum([1 for card in self.players_hand[self.cur_player] if card.rank == WIZARD])
        num_of_jesters = sum([1 for card in self.players_hand[self.cur_player] if card.rank == JESTER])
        num_of_trumps = sum([1 for card in self.players_hand[self.cur_player] if
                             card.suit == self.trump.suit and card.rank not in [JESTER, WIZARD]])

        card_played_in_trick = self.cards_played_in_trick
        players_left = self.num_players - self.player_counter - 1
        cards_left = len(self.beat_current_high_card())
        trump_color = self.trump.suit

        card_played = self.cards_played
        wizards_played = sum([1 for card in self.cards_played if card.rank == WIZARD])
        jesters_played = sum([1 for card in self.cards_played if card.rank == JESTER])
        trump_played = sum([1 for card in self.cards_played if
                            card.suit == self.trump.suit and card.rank not in [JESTER, WIZARD]])

        cur_player = self.cur_player
        tricks_left = self.players_bid[cur_player] - self.players_tricks[cur_player]
        cur_player = (cur_player + 1) % self.num_players
        tricks_left_opp1 = self.players_bid[cur_player] - self.players_tricks[cur_player]
        cur_player = (cur_player + 1) % self.num_players
        tricks_left_opp2 = self.players_bid[cur_player] - self.players_tricks[cur_player]
        cur_player = (cur_player + 1) % self.num_players
        tricks_left_opp3 = self.players_bid[cur_player] - self.players_tricks[cur_player]

        def cards_to_tensor(cards, max_size):
            tensor = torch.tensor([(card.rank, card.suit.value) for card in cards]).flatten()
            return F.pad(tensor, (0, max_size * 2 - len(tensor)))

        state_tensor = torch.cat([
            cards_to_tensor(hand, 15),
            cards_to_tensor(card_played_in_trick, 4),
            cards_to_tensor(card_played, 60),
            torch.tensor([trump_color.value]),
            torch.tensor([card_left,
                          num_of_wizards,
                          num_of_jesters,
                          num_of_trumps,
                          players_left,
                          cards_left,
                          wizards_played,
                          jesters_played,
                          trump_played,
                          tricks_left,
                          tricks_left_opp1,
                          tricks_left_opp2,
                          tricks_left_opp3]),
        ])

        return state_tensor.type(torch.long).unsqueeze(0)  # add batch dim (1, 172)
