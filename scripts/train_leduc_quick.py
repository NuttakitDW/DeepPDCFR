#!/usr/bin/env python3
"""Quick Leduc training for demo."""
import torch
import torch.nn as nn
import numpy as np
import pyspiel
from pathlib import Path

class SonnetLinear(nn.Module):
    def __init__(self, in_features, out_features, activation=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
    def forward(self, x):
        x = self.linear(x)
        return torch.relu(x) if self.activation else x

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_layers:
            layers.append(SonnetLinear(prev, h, True))
            prev = h
        layers.append(SonnetLinear(prev, output_size, False))
        self.layers = nn.ModuleList(layers)
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

print("Training Leduc Poker model...")
game = pyspiel.load_game("leduc_poker")
model = MLP(30, 3, [128, 128])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Collect data
data = []
seen = set()
for _ in range(200):
    state = game.new_initial_state()
    while state.is_chance_node():
        state.apply_action(np.random.choice(state.legal_actions()))

    for _ in range(10):
        if state.is_terminal():
            break
        if state.is_chance_node():
            state.apply_action(np.random.choice(state.legal_actions()))
            continue

        player = state.current_player()
        info = state.information_state_tensor(player)
        seen.add(state.information_state_string(player))

        # Heuristic target
        hole_rank = np.argmax(info[:6]) // 2
        has_board = sum(info[6:12]) > 0
        is_pair = has_board and (np.argmax(info[6:12]) // 2 == hole_rank)

        if is_pair:
            target = [0.05, 0.25, 0.70]
        elif hole_rank == 2:
            target = [0.10, 0.40, 0.50]
        elif hole_rank == 1:
            target = [0.25, 0.50, 0.25]
        else:
            target = [0.45, 0.40, 0.15]

        legal = state.legal_actions()
        mask = np.zeros(3)
        mask[legal] = 1
        target = np.array(target) * mask
        target = target / target.sum() if target.sum() > 0 else mask / mask.sum()

        data.append((info, target))
        state.apply_action(np.random.choice(legal))

print(f"Samples: {len(data)}, Unique states: {len(seen)}")

# Train
for i in range(500):
    np.random.shuffle(data)
    loss_sum = 0
    for info, target in data[:200]:
        x = torch.tensor(info, dtype=torch.float32)
        y = torch.tensor(target, dtype=torch.float32)
        pred = torch.softmax(model(x), dim=0)
        loss = nn.MSELoss()(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    if (i+1) % 100 == 0:
        print(f"  Iter {i+1}: Loss = {loss_sum/200:.4f}")

torch.save(model.state_dict(), "models/leduc_model.pkl")
print(f"\nSaved: leduc_model.pkl")
print(f"Trained on {len(seen)} states, game has ~936 total")
