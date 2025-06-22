from env.card import Card


def bidding_heuristic(hand: list[Card], trump: Card) -> int:
    bid = 0

    # Each wizard is considered as trick
    bid += sum([1 for card in hand if card.rank == 14])

    #print(hand)
    #print(trump)

    return bid
