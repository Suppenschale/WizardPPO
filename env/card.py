from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from env.suit import Suit

WIZARD = 14
JESTER = 0


@dataclass(frozen=True)
class Card:
    rank: int
    suit: Suit

    def __post_init__(self):
        if self.rank in [JESTER, WIZARD]:
            object.__setattr__(self, "suit", Suit.NO_SUIT)

    def is_higher_than(self, other: Card, trump: Optional[Card]):

        # other Wizard -> loose
        if other.rank == WIZARD:
            return False

        # self Wizard  -> wins (other.rank != WIZARD)
        if self.rank == WIZARD:
            return True

        # self Jester -> loose
        if self.rank == JESTER:
            return False

        # other Jester -> win (self.rank != JESTER)
        if other.rank == JESTER:
            return True

        # self trump and other no trump -> wins
        # but trump is no Jester
        if self.suit == trump.suit and other.suit != trump.suit and trump.rank != JESTER:
            return True

        # self no trump and other trump -> loose
        # but trump is no Jester
        if self.suit != trump.suit and other.suit == trump.suit and trump.rank != JESTER:
            return False

        # different suits -> loose
        if self.suit != other.suit:
            return False

        # win decided by rank
        return self.rank > other.rank
