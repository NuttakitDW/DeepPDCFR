#!/usr/bin/env python3
"""
Demo: Train and use a Kuhn Poker model

This script shows:
1. How to train a model
2. How to save it
3. How to load it
4. How to query it for strategy given your hand
"""

import torch
import torch.nn as nn
import numpy as np
import pyspiel
from scipy import stats


# Define MLP architecture (same as in deeppdcfr/deep_cfr.py)
class SonnetLinear(nn.Module):
    """Linear layer with truncated normal initialization."""
    def __init__(self, in_features, out_features, activation=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        self._initialize_weights()

    def _initialize_weights(self):
        stddev = 1.0 / np.sqrt(self.in_features)
        truncated_normal = stats.truncnorm(-2, 2, loc=0, scale=stddev)
        weight_values = truncated_normal.rvs(size=(self.out_features, self.in_features))
        self.linear.weight.data = torch.tensor(weight_values, dtype=torch.float32)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        if self.activation:
            x = torch.relu(x)
        return x


class MLP(nn.Module):
    """Multi-layer perceptron for policy/value networks."""
    def __init__(self, input_size, output_size, hidden_layers):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(SonnetLinear(prev_size, hidden_size, activation=True))
            prev_size = hidden_size
        layers.append(SonnetLinear(prev_size, output_size, activation=False))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ============================================================
# STEP 1: Setup Game and Model
# ============================================================
print("\n" + "="*60)
print("STEP 1: Setting up Kuhn Poker")
print("="*60)

game = pyspiel.load_game("kuhn_poker")
input_size = game.information_state_tensor_size()  # 11
output_size = game.num_distinct_actions()  # 2

print(f"Input size (state tensor): {input_size}")
print(f"Output size (actions): {output_size}")

# Create the neural network
network_layers = [64, 64]  # 2 hidden layers with 64 units each
model = MLP(input_size, output_size, network_layers)
print(f"\nModel architecture:")
print(model)

# ============================================================
# STEP 2: Quick Training (simplified for demo)
# ============================================================
print("\n" + "="*60)
print("STEP 2: Quick Training Demo")
print("="*60)

# For real training, use: python scripts/run.py with configs/DREAM.yaml
# Here we do a simplified training loop for demonstration

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

# Generate some training data by traversing the game tree
def collect_training_data(game, num_samples=1000):
    """Collect (state, target_policy) pairs from random playouts"""
    data = []
    for _ in range(num_samples):
        state = game.new_initial_state()

        # Deal random cards
        while state.is_chance_node():
            actions = state.legal_actions()
            state.apply_action(np.random.choice(actions))

        # Play random actions and collect states
        while not state.is_terminal():
            player = state.current_player()
            info_tensor = state.information_state_tensor(player)
            legal_actions = state.legal_actions()

            # Simple heuristic target: bet more with better cards
            card = np.argmax(info_tensor[:3])  # 0=Jack, 1=Queen, 2=King

            # Target policy based on card strength
            if card == 2:  # King
                target = [0.1, 0.9]  # Bet/Call mostly
            elif card == 1:  # Queen
                target = [0.5, 0.5]  # Mixed
            else:  # Jack
                target = [0.8, 0.2]  # Pass/Fold mostly

            data.append((info_tensor, target))

            # Random action
            state.apply_action(np.random.choice(legal_actions))

    return data

print("Collecting training data...")
training_data = collect_training_data(game, num_samples=2000)
print(f"Collected {len(training_data)} samples")

print("Training for 100 epochs...")
for epoch in range(100):
    np.random.shuffle(training_data)
    total_loss = 0

    for info_tensor, target in training_data[:500]:  # Mini-batch
        x = torch.tensor(info_tensor, dtype=torch.float32)
        y = torch.tensor(target, dtype=torch.float32)

        pred = torch.softmax(model(x), dim=0)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}: Loss = {total_loss/500:.4f}")

# ============================================================
# STEP 3: Save the Model
# ============================================================
print("\n" + "="*60)
print("STEP 3: Saving the Model")
print("="*60)

save_path = "models/demo_kuhn_model.pkl"
torch.save(model.state_dict(), save_path)
print(f"Model saved to: {save_path}")

# ============================================================
# STEP 4: Load the Model
# ============================================================
print("\n" + "="*60)
print("STEP 4: Loading the Model")
print("="*60)

# Create a new model with same architecture
loaded_model = MLP(input_size, output_size, network_layers)

# Load the weights
loaded_model.load_state_dict(torch.load(save_path, weights_only=True))
loaded_model.eval()
print("Model loaded successfully!")

# ============================================================
# STEP 5: Use Model for Strategy - Interactive Query
# ============================================================
print("\n" + "="*60)
print("STEP 5: Querying Model for Strategy")
print("="*60)

def get_strategy(model, game, card: str, history: str = ""):
    """
    Get strategy from model given your card and betting history.

    Args:
        model: Trained neural network
        game: pyspiel game
        card: 'J', 'Q', or 'K' (Jack, Queen, King)
        history: betting history string like '', 'p', 'b', 'pb', etc.
                 p = pass/check, b = bet/call

    Returns:
        Dict of action -> probability
    """
    # Map card to action
    card_map = {'J': 0, 'Q': 1, 'K': 2}
    if card.upper() not in card_map:
        raise ValueError("Card must be J, Q, or K")

    card_action = card_map[card.upper()]

    # Create game state
    state = game.new_initial_state()

    # Deal cards - we need to figure out which player we are
    # based on the history length
    is_player_0 = len(history) % 2 == 0 if history else True

    if is_player_0:
        state.apply_action(card_action)  # We get our card
        state.apply_action((card_action + 1) % 3)  # Opponent gets different card
        player = 0
    else:
        state.apply_action((card_action + 1) % 3)  # Opponent gets different card
        state.apply_action(card_action)  # We get our card
        player = 1

    # Apply betting history
    action_map = {'p': 0, 'b': 1, 'P': 0, 'B': 1}
    for action_char in history:
        if action_char in action_map:
            state.apply_action(action_map[action_char])

    if state.is_terminal():
        return {"Game is over": 1.0}

    # Get information state tensor
    info_tensor = state.information_state_tensor(player)

    # Query model
    with torch.no_grad():
        x = torch.tensor(info_tensor, dtype=torch.float32)
        logits = model(x)
        probs = torch.softmax(logits, dim=0).numpy()

    # Get legal actions and their probabilities
    legal_actions = state.legal_actions()
    action_names = {0: "Pass/Fold", 1: "Bet/Call"}

    strategy = {}
    for action in legal_actions:
        strategy[action_names[action]] = float(probs[action])

    return strategy, info_tensor

# Demo queries
print("\n" + "-"*60)
print("EXAMPLE QUERIES:")
print("-"*60)

examples = [
    ('J', '', "You have JACK, first to act"),
    ('Q', '', "You have QUEEN, first to act"),
    ('K', '', "You have KING, first to act"),
    ('J', 'b', "You have JACK, opponent BET"),
    ('Q', 'b', "You have QUEEN, opponent BET"),
    ('K', 'b', "You have KING, opponent BET"),
    ('J', 'p', "You have JACK, opponent PASSED"),
    ('K', 'pb', "You have KING, you passed, opponent BET"),
]

for card, history, description in examples:
    strategy, tensor = get_strategy(loaded_model, game, card, history)
    print(f"\n{description}")
    print(f"  Card: {card}, History: '{history}' (empty=first to act)")
    print(f"  Input tensor: {[int(x) for x in tensor[:6]]}...")
    print(f"  Strategy: ", end="")
    for action, prob in strategy.items():
        print(f"{action}={prob:.1%}  ", end="")
    print()

# ============================================================
# STEP 6: Interactive Mode
# ============================================================
print("\n" + "="*60)
print("STEP 6: Interactive Query Mode")
print("="*60)
print("""
You can now query the model yourself!

Usage: Enter your card (J/Q/K) and betting history

Examples:
  - 'K'     → You have King, first to act
  - 'Q b'   → You have Queen, opponent bet
  - 'J pb'  → You have Jack, you passed, opponent bet

History codes: p=pass/check, b=bet/call

Type 'quit' to exit.
""")

def interactive_mode():
    while True:
        try:
            user_input = input("\nEnter (card [history]): ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            parts = user_input.split()
            if not parts:
                continue

            card = parts[0].upper()
            history = parts[1] if len(parts) > 1 else ""

            strategy, tensor = get_strategy(loaded_model, game, card, history)

            print(f"\n  Your card: {card}")
            print(f"  History: '{history}'")
            print(f"  Input tensor: {[round(x, 1) for x in tensor]}")
            print(f"\n  RECOMMENDED STRATEGY:")
            for action, prob in strategy.items():
                bar = "█" * int(prob * 20)
                print(f"    {action:12s}: {prob:6.1%} {bar}")

        except ValueError as e:
            print(f"  Error: {e}")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    # Uncomment to enable interactive mode:
    # interactive_mode()
    print("\nTo run interactive mode, uncomment the last line or run:")
    print("  python scripts/demo_kuhn_poker.py")
    print("\nDemo complete!")
