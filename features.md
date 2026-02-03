# Model Features Specification

## Overview

This document defines the input features for the postflop poker solver neural network model.

## Feature Table

| Feature | Value | Method | Encoding | Size |
|---------|-------|--------|----------|------|
| Stack | 30bb | Fixed | stack / 200 | 1 |
| Pot | 3-25bb | Random sample | pot / 100 | 1 |
| SPR | Derived | pot ÷ stack | log(spr+1) / 4 | 1 |
| Flop | 0-1754 | Random + Embedding | Embedding lookup | 32 |
| Turn | 0-51 | Isomorphic + Embedding | Embedding lookup | 8 |
| River | 0-51 | Isomorphic + Embedding | Embedding lookup | 8 |
| OOP Range | Preflop range | Random archetype + Autoencoder | Range → 1326 sparse → 64 | 64 |
| IP Range | Preflop range | Random archetype + Autoencoder | Range → 1326 sparse → 64 | 64 |
| Bet Facing | % of pot | Inside tree | bet / pot | 1 |
| Street | F/T/R | Inside tree | One-hot [1,0,0] | 3 |
| Who Acts | OOP/IP | Inside tree | Binary [0 or 1] | 1 |
| Player Hand | 0-1325 | Inside tree | Embedding lookup | 16 |
| Pot Odds | Derived | bet / (pot + bet) | Float | 1 |
| Stack Remaining | Derived | (stack - bet) / 200 | Float | 1 |

## Total Input Size

```
Stack:           1
Pot:             1
SPR:             1
Flop:           32
Turn:            8
River:           8
OOP Range:      64
IP Range:       64
Bet Facing:      1
Street:          3
Who Acts:        1
Player Hand:    16
Pot Odds:        1
Stack Remaining: 1
─────────────────────
Total:         202 dims
```

## Feature Details

### Root Node Features (Scenario Definition)

These features define a unique scenario:

| Feature | Description |
|---------|-------------|
| Stack | Effective stack size in big blinds |
| Pot | Starting pot size in big blinds |
| Flop | Canonical flop ID (0-1754 after isomorphism) |
| OOP Range | Out-of-position player's hand range |
| IP Range | In-position player's hand range |

### Derived Features

Calculated from other features:

| Feature | Formula |
|---------|---------|
| SPR | stack / pot |
| Pot Odds | bet / (pot + bet) |
| Stack Remaining | (stack - bet) / 200 |

### Inside Tree Features

These change during game tree traversal:

| Feature | Description |
|---------|-------------|
| Turn | Turn card (if dealt) |
| River | River card (if dealt) |
| Bet Facing | Current bet to call as % of pot |
| Street | Current street (Flop/Turn/River) |
| Who Acts | Current player (OOP=0, IP=1) |
| Player Hand | Specific hand being evaluated |

## Encoding Methods

### Normalization

```python
stack_encoded = stack / 200      # Range: 0-1 for 0-200bb
pot_encoded = pot / 100          # Range: 0-1 for 0-100bb
spr_encoded = log(spr + 1) / 4   # Log scale for SPR
bet_encoded = bet / pot          # Fraction of pot
```

### Embedding

Learned lookup tables that convert IDs to dense vectors:

```python
flop_embedding = nn.Embedding(1755, 32)   # 1755 canonical flops
turn_embedding = nn.Embedding(52, 8)       # 52 cards
river_embedding = nn.Embedding(52, 8)      # 52 cards
hand_embedding = nn.Embedding(1326, 16)    # 1326 hand combos
```

### Autoencoder

Range compression from 1326 weights to 64 dimensions:

```python
# Range string → 1326 vector → Autoencoder → 64 dims
range_vector = parse_range_to_1326(range_string)
range_embedding = autoencoder.encode(range_vector)
```

### One-Hot

```python
street_onehot = {
    "flop":  [1, 0, 0],
    "turn":  [0, 1, 0],
    "river": [0, 0, 1],
}
```

## Isomorphism

### Flop Isomorphism

22,100 raw flops reduced to 1,755 canonical flops via suit isomorphism.

Example: A♠K♠7♥ ≈ A♥K♥7♠ ≈ A♦K♦7♣ (same canonical flop)

### Turn/River Isomorphism

Isomorphism relative to board suits:
- Cards of equivalent suits (relative to flop) are combined
- Reduces computation without losing strategic information

## Training Strategy

### Random Sampling

```python
scenario = {
    "stack": 30,                    # Fixed for 30bb model
    "pot": random(3, 25),           # Random
    "flop": random(0, 1754),        # Random canonical flop
    "range_pair": random(0, 9),     # Random archetype
}
```

### Continuous Training

1. Generate random scenarios
2. Solve with DCFR+
3. Add to training data
4. Train/update model
5. Save checkpoint
6. Repeat

### Targeted Training

When specific spots show high deviation from GTO:
1. Identify weak spots (e.g., monotone boards, 3bet pots)
2. Generate focused scenarios for those spots
3. Retrain model on combined data

## Model Expansion

### Stack Expansion

Start with 30bb model, then expand:

```
Model_30bb → Train until good
Model_50bb → Train separately
Model_100bb → Train separately
```

Or later: Combine all data → Train universal model

### Feature Expansion

Add features if specific decisions are weak:

```
v1: Current features
v2: Add new features if needed
v3: Continue improving
```
