#!/usr/bin/env python3
"""
No-Limit Poker Solver with NUMERIC features.

This demonstrates:
- Numeric inputs (pot size, stack size, bet amounts)
- Variable action space (different bet sizes)
- How real poker solvers encode state

Usage:
    python scripts/nolimit_solver.py --train
    python scripts/nolimit_solver.py --query
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import pyspiel
from pathlib import Path


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


# Card values
RANKS = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
         '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
SUITS = {'c': 0, 'd': 1, 'h': 2, 's': 3}


def extract_numeric_features(state, player=0):
    """
    Extract NUMERIC features from poker state.

    Returns vector with actual numbers, not one-hot:
    - pot_ratio: pot / starting_stack
    - stack_ratio: my_stack / starting_stack
    - opponent_stack_ratio: opp_stack / starting_stack
    - card_strength: 0-1 value based on card rank
    - position: 0 or 1
    - round: 0 or 1
    - facing_bet: 0 or 1
    - bet_to_call_ratio: amount_to_call / pot
    """
    info_str = state.information_state_string(player)

    # Parse info string: [Round 0][Player: 0][Pot: 200][Money: 950 900][Private: 2c]...
    features = {}

    # Extract values using simple parsing
    import re

    round_match = re.search(r'\[Round (\d+)\]', info_str)
    features['round'] = int(round_match.group(1)) if round_match else 0

    pot_match = re.search(r'\[Pot: (\d+)\]', info_str)
    pot = int(pot_match.group(1)) if pot_match else 0

    money_match = re.search(r'\[Money: (\d+) (\d+)\]', info_str)
    if money_match:
        stacks = [int(money_match.group(1)), int(money_match.group(2))]
    else:
        stacks = [1000, 1000]

    card_match = re.search(r'\[Private: (\w+)\]', info_str)
    if card_match:
        card = card_match.group(1)
        if len(card) >= 2:
            rank = RANKS.get(card[0].upper(), 0)
            card_strength = rank / 12.0  # Normalize to 0-1
        else:
            card_strength = 0.5
    else:
        card_strength = 0.5

    # Build feature vector (all NUMERIC, not one-hot)
    starting_stack = 1000
    my_stack = stacks[player]
    opp_stack = stacks[1 - player]

    feature_vector = [
        pot / starting_stack,              # Pot ratio (0-2+)
        my_stack / starting_stack,         # My stack ratio (0-1)
        opp_stack / starting_stack,        # Opponent stack ratio (0-1)
        card_strength,                     # Card strength (0-1)
        float(player),                     # Position (0 or 1)
        features['round'] / 2.0,           # Round (0 or 0.5)
        1.0 if pot > 200 else 0.0,        # Facing bet indicator
        min(pot / max(my_stack, 1), 2.0),  # Pot-to-stack ratio
    ]

    return np.array(feature_vector, dtype=np.float32)


def get_bet_sizes(pot, stack, min_bet=100):
    """Generate discrete bet size options as fractions of pot."""
    sizes = []

    # Standard bet sizes as pot fractions
    fractions = [0.33, 0.5, 0.75, 1.0, 1.5, 2.0]

    for frac in fractions:
        bet = int(pot * frac)
        if min_bet <= bet <= stack:
            sizes.append(bet)

    # Add all-in
    if stack > min_bet:
        sizes.append(stack)

    return sorted(set(sizes))


class NoLimitPokerSolver:
    """
    Poker solver with numeric features and variable bet sizing.

    Input features (8 dimensions):
        [pot_ratio, my_stack, opp_stack, card_strength, position, round, facing_bet, pot_to_stack]

    Output (6 dimensions):
        [fold, check/call, bet_small, bet_medium, bet_large, all_in]
    """

    ACTION_NAMES = ['Fold', 'Check/Call', 'Bet 33%', 'Bet 50%', 'Bet 75%', 'Bet 100%+']

    def __init__(self):
        self.input_size = 8   # Numeric features
        self.output_size = 6  # Action buckets
        self.model = MLP(self.input_size, self.output_size, [64, 64])
        self.model_path = Path(__file__).parent.parent / "models" / "nolimit_model.pkl"

    def train(self, num_iterations=1000):
        """Train the model with numeric features."""
        print("Training No-Limit Poker Solver...")
        print(f"Input: {self.input_size} NUMERIC features")
        print(f"Output: {self.output_size} action buckets")
        print()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Generate training data
        data = []

        for _ in range(2000):
            # Random game state
            pot = np.random.randint(100, 2000)
            my_stack = np.random.randint(100, 1000)
            opp_stack = np.random.randint(100, 1000)
            card_strength = np.random.random()
            position = np.random.randint(0, 2)
            round_num = np.random.randint(0, 2)
            facing_bet = np.random.random() > 0.5

            features = np.array([
                pot / 1000,
                my_stack / 1000,
                opp_stack / 1000,
                card_strength,
                float(position),
                round_num / 2.0,
                float(facing_bet),
                min(pot / max(my_stack, 1), 2.0),
            ], dtype=np.float32)

            # Heuristic target based on card strength and pot odds
            if card_strength > 0.8:  # Strong hand
                target = [0.0, 0.1, 0.1, 0.2, 0.3, 0.3]  # Bet big
            elif card_strength > 0.5:  # Medium hand
                target = [0.1, 0.3, 0.3, 0.2, 0.1, 0.0]  # Bet medium
            elif card_strength > 0.3:  # Weak hand
                target = [0.2, 0.5, 0.2, 0.1, 0.0, 0.0]  # Check/call mostly
            else:  # Very weak
                target = [0.4, 0.4, 0.1, 0.1, 0.0, 0.0]  # Fold or check

            # Adjust for facing bet
            if facing_bet and card_strength < 0.4:
                target = [0.6, 0.3, 0.1, 0.0, 0.0, 0.0]  # Fold more

            target = np.array(target, dtype=np.float32)
            target = target / target.sum()

            data.append((features, target))

        print(f"Training on {len(data)} samples...")

        for iteration in range(num_iterations):
            np.random.shuffle(data)
            total_loss = 0

            for features, target in data[:256]:
                x = torch.tensor(features)
                y = torch.tensor(target)

                logits = self.model(x)
                probs = torch.softmax(logits, dim=0)
                loss = nn.MSELoss()(probs, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (iteration + 1) % 200 == 0:
                print(f"  Iter {iteration + 1}: Loss = {total_loss / 256:.4f}")

        torch.save(self.model.state_dict(), self.model_path)
        print(f"\nSaved: {self.model_path}")

    def load(self):
        """Load trained model."""
        if not self.model_path.exists():
            print("No model found. Run with --train first.")
            sys.exit(1)
        self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
        self.model.eval()

    def solve(self, pot: int, my_stack: int, opp_stack: int,
              card_strength: float, position: int = 0,
              round_num: int = 0, facing_bet: bool = False):
        """
        Query solver with NUMERIC inputs.

        Args:
            pot: Current pot size in chips
            my_stack: Your remaining stack
            opp_stack: Opponent's remaining stack
            card_strength: 0-1 (0=worst, 1=nuts)
            position: 0=first to act, 1=in position
            round_num: 0=preflop, 1=postflop
            facing_bet: True if opponent bet
        """
        features = np.array([
            pot / 1000,
            my_stack / 1000,
            opp_stack / 1000,
            card_strength,
            float(position),
            round_num / 2.0,
            float(facing_bet),
            min(pot / max(my_stack, 1), 2.0),
        ], dtype=np.float32)

        with torch.no_grad():
            x = torch.tensor(features)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=0).numpy()

        print(f"=== NO-LIMIT POKER SOLVER ===")
        print(f"Input (NUMERIC):")
        print(f"  Pot:           {pot} chips")
        print(f"  Your stack:    {my_stack} chips")
        print(f"  Opponent:      {opp_stack} chips")
        print(f"  Card strength: {card_strength:.0%}")
        print(f"  Position:      {'In position' if position else 'Out of position'}")
        print(f"  Round:         {'Postflop' if round_num else 'Preflop'}")
        print(f"  Facing bet:    {'Yes' if facing_bet else 'No'}")
        print()
        print(f"Feature vector: {[f'{x:.2f}' for x in features]}")
        print()
        print("Strategy:")

        for i, (name, prob) in enumerate(zip(self.ACTION_NAMES, probs)):
            bar = "█" * int(prob * 20)
            print(f"  {name:12s}: {prob:5.1%} {bar}")


def interactive():
    """Interactive query mode."""
    solver = NoLimitPokerSolver()
    solver.load()

    print("\n=== NO-LIMIT POKER SOLVER ===")
    print("Enter numeric values to query strategy.\n")

    while True:
        try:
            print("-" * 40)
            pot = int(input("Pot size (chips): "))
            my_stack = int(input("Your stack: "))
            opp_stack = int(input("Opponent stack: "))
            card_strength = float(input("Card strength (0-1): "))
            facing_bet = input("Facing bet? (y/n): ").lower() == 'y'

            print()
            solver.solve(pot, my_stack, opp_stack, card_strength,
                        facing_bet=facing_bet)
            print()

        except ValueError as e:
            print(f"Invalid input: {e}")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    solver = NoLimitPokerSolver()

    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        solver.train()
    elif len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive()
    else:
        # Demo query
        solver.load() if solver.model_path.exists() else solver.train()
        if solver.model_path.exists():
            solver.load()

        print("\nExample queries:\n")

        # Query 1: Small pot, strong hand
        solver.solve(pot=200, my_stack=800, opp_stack=800,
                    card_strength=0.9, facing_bet=False)
        print()

        # Query 2: Big pot, weak hand, facing bet
        solver.solve(pot=500, my_stack=500, opp_stack=600,
                    card_strength=0.2, facing_bet=True)
        print()

        # Query 3: Medium situation
        solver.solve(pot=300, my_stack=700, opp_stack=700,
                    card_strength=0.6, facing_bet=False)
