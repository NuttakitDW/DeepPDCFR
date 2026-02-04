# Suit Isomorphism and Hand Transformation

## Overview

Suit isomorphism reduces the number of unique boards the model needs to learn by treating strategically equivalent boards as the same. This document explains how isomorphism works and why **hand transformation** is critical when using it.

## The Problem: Too Many Boards

Without isomorphism:
- Flops: 22,100 unique combinations
- Turns: 22,100 × 48 = 1,060,800
- Rivers: 22,100 × 48 × 47 = 49,857,600

With isomorphism:
- Flops: **1,755** canonical combinations (12x reduction)
- Much smaller input space for model to learn

## What is Suit Isomorphism?

Suits in poker have no inherent ranking (hearts ≠ better than spades). Only **relative** suit relationships matter:

```
These boards are strategically IDENTICAL:
  As 7s 2c  (spade flush possible)
  Ah 7h 2c  (heart flush possible)
  Ad 7d 2c  (diamond flush possible)
  Ac 7c 2d  (club flush possible)

All have: "two cards of same suit + one different suit"
```

We pick ONE canonical form to represent all of them.

## Canonical Form Convention

Standard convention (most solvers):
1. First suit that appears → Hearts (h)
2. Second suit → Diamonds (d)
3. Third suit → Clubs (c)
4. Fourth suit → Spades (s)

### Examples

```
Original         Canonical        Swap Rule
─────────────────────────────────────────────
As 7s 2c    →    Ah 7h 2d        s→h, c→d
Kd Qd 5h    →    Kh Qh 5d        d→h, h→d
Jc 8c 4c    →    Jh 8h 4h        c→h (monotone)
Ac Kd Qh    →    Ah Kd Qc        c→h, h→c (rainbow)
```

## Critical: Hand Transformation

### The Problem

If you transform the board but NOT the hands, suit relationships break:

```
Original scenario:
  Board: As 7s 2c
  Hand:  AsKs = FLUSH DRAW ✓

After board isomorphism (s→h):
  Board: Ah 7h 2d
  Hand:  AsKs = NO FLUSH DRAW ✗ (wrong!)
```

### The Solution

Apply the SAME suit swap to hands:

```
Board transform: s→h, c→d
Hand transform:  AsKs → AhKh

Now:
  Board: Ah 7h 2d
  Hand:  AhKh = FLUSH DRAW ✓ (correct!)
```

## Complete Workflow

### Step 1: Compute Canonical Board + Swap Map

```python
def to_canonical(board):
    """Convert board to canonical form, return swap map."""
    suits_seen = []
    canonical_suits = ['h', 'd', 'c', 's']
    swap_map = {}

    for card in board:
        suit = card[1]
        if suit not in suits_seen:
            suits_seen.append(suit)
            swap_map[suit] = canonical_suits[len(suits_seen) - 1]

    canonical_board = [(card[0], swap_map.get(card[1], card[1]))
                       for card in board]
    return canonical_board, swap_map

# Example
board = [('A', 's'), ('7', 's'), ('2', 'c')]
canonical, swap = to_canonical(board)
# canonical = [('A', 'h'), ('7', 'h'), ('2', 'd')]
# swap = {'s': 'h', 'c': 'd'}
```

### Step 2: Transform Hands (for model input ranges)

```python
def transform_hand(hand, swap_map):
    """Apply swap map to a hand."""
    return tuple((rank, swap_map.get(suit, suit))
                 for rank, suit in hand)

# Example
hand = (('A', 's'), ('K', 's'))  # AsKs
transformed = transform_hand(hand, swap)
# transformed = (('A', 'h'), ('K', 'h'))  # AhKh
```

### Step 3: Query Model in Canonical Space

```python
def query_solver(board, situation):
    # 1. Canonicalize board
    canonical_board, swap_map = to_canonical(board)

    # 2. Build input features
    input_features = build_features(canonical_board, situation)

    # 3. Model outputs strategy for all 1326 hands
    #    (hands are in CANONICAL space)
    canonical_output = model(input_features)  # [1326 × 3]

    return canonical_output, swap_map
```

### Step 4: Reverse Transform for Display

```python
def reverse_swap(swap_map):
    """Create reverse mapping."""
    return {v: k for k, v in swap_map.items()}

def display_results(canonical_output, swap_map):
    """Convert canonical hands back to original suits."""
    reverse_map = reverse_swap(swap_map)

    results = {}
    for hand_idx in range(1326):
        canonical_hand = index_to_hand(hand_idx)
        original_hand = transform_hand(canonical_hand, reverse_map)
        results[original_hand] = canonical_output[hand_idx]

    return results
```

## Full Example

User queries: **"What's the strategy on As7s2c?"**

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT                                                       │
│   Board: As 7s 2c                                          │
│   Question: Strategy for all hands?                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Canonicalize                                        │
│   Board: As 7s 2c → Ah 7h 2d                               │
│   Swap:  s→h, c→d                                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Model Query (canonical space)                       │
│   Input:  [Ah 7h 2d embedding] + situation                 │
│   Output: 1326 strategies (canonical hands)                 │
│                                                             │
│   AhKh → [Fold: 5%, Call: 30%, Raise: 65%]  (flush draw)  │
│   AsKs → [Fold: 8%, Call: 40%, Raise: 52%]  (no draw)     │
│   AdKd → [Fold: 8%, Call: 42%, Raise: 50%]  (backdoor)    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Reverse Transform (h→s, d→c)                        │
│                                                             │
│   Canonical    Original    Strategy                         │
│   ─────────────────────────────────────                     │
│   AhKh     →   AsKs        [5%, 30%, 65%]   ← flush draw   │
│   AsKs     →   AhKh        [8%, 40%, 52%]   ← no draw      │
│   AdKd     →   AcKc        [8%, 42%, 50%]   ← backdoor     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ DISPLAY TO USER                                             │
│                                                             │
│   Board: As 7s 2c                                          │
│   ──────────────────────────────────────                    │
│   AsKs  [5%, 30%, 65%]  ← spade flush draw                 │
│   AhKh  [8%, 40%, 52%]  ← no flush draw                    │
│   AcKc  [8%, 42%, 50%]  ← backdoor club                    │
│   AdKd  [8%, 42%, 50%]  ← backdoor diamond                 │
└─────────────────────────────────────────────────────────────┘
```

## Key Points

| Concept | Description |
|---------|-------------|
| Why isomorphism | Reduce 22,100 flops → 1,755 (12x smaller) |
| Canonical form | Standard representation for equivalent boards |
| Swap map | Records how suits were remapped |
| Hand transform | **MUST** apply same swap to hands |
| Reverse transform | Convert back to original suits for display |

## Common Mistakes

### ❌ Wrong: Transform board only

```python
# BUG: hands don't match board
canonical_board = to_canonical(board)
output = model(canonical_board)
display(output)  # AsKs shows wrong strategy!
```

### ✓ Correct: Transform both, reverse for display

```python
canonical_board, swap_map = to_canonical(board)
output = model(canonical_board)  # canonical hands
display(reverse_transform(output, swap_map))  # original hands
```

## Implementation Notes

1. **Training data**: Generate in canonical space only
2. **Ranges**: If input includes ranges, transform those too
3. **Turn/River**: Continue using same swap map from flop
4. **Caching**: Can cache canonical board embeddings (only 1,755 flops)

## References

- Johanson, M. (2007). Robust Strategies and Counter-Strategies: Building a Champion Level Computer Poker Player
- Brown, N. & Sandholm, T. (2019). Superhuman AI for multiplayer poker (Pluribus)
