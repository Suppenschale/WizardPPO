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
        # self Wizard -> wins if other no Wizard
        if self.rank == 14:
            return self.rank > other.rank

        # other Wizard -> loose
        if other.rank == 14:
            return False

        # self Jester -> loose
        if self.rank == 0:
            return False

        # self trump and other no trump -> wins
        # but trump is no Jester
        if trump and self.suit == trump.suit and other.suit != trump.suit and trump.rank != 0:
            return True

        # self no trump and other trump -> loose
        # but trump is no Jester
        if trump and self.suit != trump.suit and other.suit == trump.suit and trump.rank != 0:
            return False

        # different suits -> loose
        if self.suit != other.suit:
            return False

        # win decided by rank
        return self.rank > other.rank
