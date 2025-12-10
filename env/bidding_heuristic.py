import numpy as np

from env.card import Card
from env.environment import JESTER
from env.environment import WIZARD


def bidding_heuristic(hand: list[Card], trump: Card) -> int:
    bid = 0

    for card in hand:

        # For every Wizard in hand increase the bid
        if card.rank == WIZARD:
            bid += 1

        # For every trump in hand increase the bid
        if trump and card.suit == trump.suit and card.rank not in [JESTER, WIZARD]:
            bid += 1

        # For every 13 in hand increase the bid
        if trump and card.rank == 13 and card.suit != trump.suit:
            bid += 1

        # In the last round (no trump), count all cards >= 10
        if not trump and card.rank >= 10 and card.rank != WIZARD:
            bid += 1

    return np.clip(bid , 0, len(hand)) #+ np.random.randint(0, 0)
