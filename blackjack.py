import random

suits = ('Hearts', 'Diamonds', 'Spades', 'Clubs')
ranks = ('Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King', 'Ace')
values = {'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5, 'Six': 6, 'Seven': 7, 'Eight': 8, 'Nine': 9, 'Ten': 10,
          'Jack': 10,
          'Queen': 10, 'King': 10, 'Ace': 11}


# CLASS DEFINTIONS:

class Card:

    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return self.rank + ' of ' + self.suit


class Deck:

    def __init__(self):
        self.deck = []
        for suit in suits:
            for rank in ranks:
                self.deck.append(Card(suit, rank))  # Create 52 Instance of Card

    def show(self):
        for d in self.deck:
            print(d)

    def __str__(self):
        all_cards = ''
        i = 0
        for card in self.deck:
            i += 1
            all_cards += str(i) + ') ' + card.__str__() + '\n'
        return all_cards

    def shuffle_cards(self):
        random.shuffle(self.deck)  # shucfle deck in place

    def deal(self):
        single_card = self.deck.pop()
        return single_card


class Hand:

    def __init__(self):
        self.cards = []
        self.value = 0
        self.aces = 0  # To keep track of aces

    def add_card(self, card):
        self.cards.append(card)  # Appending Card Objects to a list
        self.value += values[card.rank]

        if card.rank == 'Ace':  # Add Aces
            self.aces += 1

    def adjust_for_ace(self):
        while self.value > 21 and self.aces:  # continue while value>21 and cantain at least 1 ace
            self.value -= 10  # Reduces by 10
            self.aces -= 1  # Reduce Aces until 0


class Chips:

    def __init__(self, total):
        self.total = total  # This can be set to a default value or supplied by a user input
        self.bet = 0

    def win_bet(self):
        self.total += self.bet
        return self.total

    def lose_bet(self):
        self.total -= self.bet
        return self.total


# FUNCTION DEFINITIONs

def take_bet(chips, name):
    while True:
        try:
            chips.bet = int(input(f'How many chips would you like to bet {name}?:'))
        except:
            print('Invalid Input Please try again!')
            continue
        else:
            if chips.bet > chips.total:
                print("Sorry, your bet can't exceed", chips.total)
            else:
                break


def hit(deck, hand):
    hand.add_card(deck.deal())  # Release a card from deck and add to player's hand
    hand.adjust_for_ace()


def hit_or_stand(deck, hand, name):
    global playing  # to control an upcoming while loop
    global stand  # to control an upcoming while loop

    while True:
        x = input(f"\nWould you like to Hit or Stand {name}? Enter 'h' or 's': ").lower()
        if x == 'h':
            hit(deck, hand)

        elif x == 's':
            print(f"\n{name} stands. Now next player turn.")
            stand = 's'
            playing.pop()
            break

        else:
            print("Sorry, please try again.")
            continue
        break


def show_myhand(hand, name):
    print('\nMy Hand: ', *hand[name][0].cards, sep='\n')
    print('   My Hand Value: ', hand[name][0].value)


def show_some(players, dealer):
    print("\n|| DEALER'S HAND ||")
    print("  <card hidden>")
    # print('' + dealer.cards[1].__str__())  OR [with (+) we have to call .__str__() explicitly]
    print('  ', dealer.cards[1])  # [with (,) .__str__() is called implicitly] ())[0][0].value

    print("\n|| PLAYER'S HAND ||\n")
    for name in players:
        i = 0
        print(name + "'s Hand")

        while i < len(players[name][0].cards):
            print('\t' + players[name][0].cards[i].__str__())
            i += 1
        print('\tCards Value:', players[name][0].value)
        print()


def show_all(players, dealer):
    print("\n>>>>DEALER'S HAND:", *dealer.cards, sep='\n ')
    print(" Cards Value =", dealer.value)

    print("\n>>>>PLAYER'S HAND:\n")
    for name in players:
        i = 0
        print(name + "'s Hand")

        while i < len(players[name][0].cards):
            print('\t' + players[name][0].cards[i].__str__())
            i += 1

        print('\tCards Value:', players[name][0].value)
        print()


def player_busts(player_chips, name):
    print(f'\n=== {name} IS BUSTED ===')
    print('   Spendalbe Amount: ', player_chips.lose_bet())


def player_wins(player_chips, name):
    print(f'\n=== {name} Won ===')
    print('   Total Amount: ', player_chips.win_bet())


def dealer_busts(players):
    print('\n=== Dealer Busted ===')

    for name in players:
        print(f'{name} WON!\n   Total amount: {players[name][1].win_bet()}')


def dealer_wins(player_chips, name):
    print(f'\n>>> Dealer Won! From {name} <<<')
    print(f'     Spendable Amount: {player_chips.lose_bet()}\n')


def push(p_value, name, d_value):
    print(f"\n=== Dealer & {name} Tie. Its a PUSH ===")
    print(f'\nPlayer Value: {p_value} & Dealer Value: {d_value}\n')


# GAMEPLAY!!

while True:

    print('\n<< Welcome to BLACKJACK! >> \n\t>> Get as close to 21 as you can without going over!\n\
    Dealer hits until he reaches 17. Aces count as 1 or 11 <<')

    # Create & shuffle the deck

    deck = Deck()
    deck.shuffle_cards()

    dealer_hand = Hand()
    dealer_hand.add_card(deck.deal())
    dealer_hand.add_card(deck.deal())

    while True:
        try:
            n = int(input('Enter number of players: '))
        except:
            print('Invalid.... Please Try Again!')
            continue
        else:
            break

    playing = [True for i in range(n)]  # To control upcoming while loop
    players_hand = {}

    for i in range(n):
        players_hand[input('Enter Your Name: ').capitalize()] = (Hand(), Chips(100))

    for name in players_hand:
        players_hand[name][0].add_card(deck.deal())
        players_hand[name][0].add_card(deck.deal())
        take_bet(players_hand[name][1], name)

    show_some(players_hand, dealer_hand)

    ph = dict(players_hand)
    busted_players = []

    while True:
        try:
            for name in players_hand:
                stand = ''
                while True in playing and stand != 's':
                    hit_or_stand(deck, players_hand[name][0], name)  # Passing Player's Hand to 'H' or 'S'
                    show_myhand(players_hand, name)

                    if players_hand[name][0].value > 21:
                        player_busts(players_hand[name][1], name)  # Pass the Busted Player Chips
                        busted_players.append(name)  # Store busted Players Names
                        # players_hand.pop(name)                         # Pop the Busted Player for Dict
                        playing.pop()  # Pop True from list
                        break

        except RuntimeError:
            pass
        else:
            break

    while dealer_hand.value < 17 and True in playing:  # If any of the player is still alive Play the Dealer's Hand
        hit(deck, dealer_hand)

    show_all(players_hand, dealer_hand)

    for name in players_hand:

        if dealer_hand.value > 21:
            dealer_busts(players_hand)
            break

        if players_hand[name][0].value <= 21:

            if dealer_hand.value > players_hand[name][0].value:
                dealer_wins(players_hand[name][1], name)

            elif dealer_hand.value < players_hand[name][0].value:
                player_wins(players_hand[name][1], name)

            else:
                push(players_hand[name][0].value, name, dealer_hand.value)
        else:
            pass

    if input("Do you want to play again? Enter 'y' or 'n' ") == 'y':
        continue
    else:
        print(f'\n\t\t****** THANKS FOR PLAYING: {[k for k in ph]} ******')
        break
