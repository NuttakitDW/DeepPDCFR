# Training Sampling Strategy

## Philosophy

Deep PDCFR+ is **self-play reinforcement learning**, not supervised learning. The algorithm explores and learns optimal play through random scenario generation.

**Goal**: Train an ENGINE that can solve ANY postflop input with ~99% accuracy.

| Aspect | Traditional Solver | Deep PDCFR+ |
|--------|-------------------|-------------|
| Training data | N/A | Random exploration |
| Ranges | Must be exact | Can be random |
| Output | Lookup table | Learned solver |
| Generalization | None | Any input |

## Root Node Inputs (Sampled Per Episode)

These are randomly sampled at the start of each training episode:

| Input | Sampling | Range | Encoding | Size |
|-------|----------|-------|----------|------|
| Stack | Uniform | 10-200bb | stack / 200 | 1 |
| Pot | Uniform | 2-100bb | pot / 100 | 1 |
| SPR | Derived | stack / pot | log(spr+1) / 4 | 1 |
| Flop | Random 3 cards | 1755 canonical | Embedding | 32 |
| Turn | Random 1 card | 49 remaining | Embedding | 8 |
| River | Random 1 card | 48 remaining | Embedding | 8 |
| OOP Range | Uniform or random | 1326 weights | Autoencoder → 64 | 64 |
| IP Range | Uniform or random | 1326 weights | Autoencoder → 64 | 64 |

## Inside Tree Inputs (Change During Traversal)

These change as the algorithm traverses the game tree:

| Input | Description | Encoding | Size |
|-------|-------------|----------|------|
| Street | Flop/Turn/River | One-hot [1,0,0] | 3 |
| Who Acts | OOP or IP | Binary [0 or 1] | 1 |
| Player Hand | Hand being evaluated | Embedding | 16 |
| Bet Facing | Current bet to call | bet / pot | 1 |
| Pot Odds | Derived | bet / (pot + bet) | 1 |
| Stack Remaining | Derived | (stack - bet) / 200 | 1 |

## Total Input Size: 202 dimensions

```
Stack:              1
Pot:                1
SPR:                1
Flop Embedding:    32
Turn Embedding:     8
River Embedding:    8
OOP Range:         64
IP Range:          64
Bet Facing:         1
Street:             3
Who Acts:           1
Player Hand:       16
Pot Odds:           1
Stack Remaining:    1
─────────────────────
Total:            202
```

## Range Sampling Options

The algorithm can learn with any of these approaches:

### Option A: Uniform (Simplest)
```python
oop_range = [1.0] * 1326  # All hands equally likely
ip_range = [1.0] * 1326
```

### Option B: Random Weights
```python
oop_range = [random.random() for _ in range(1326)]
ip_range = [random.random() for _ in range(1326)]
```

### Option C: Archetype-Based (Optional)
```python
# Sample from predefined archetypes for more realistic distribution
archetype = random.choice(["tight", "medium", "wide"])
oop_range = load_archetype(archetype)
```

**Recommendation**: Uniform or random weights. The network learns relationships regardless of range distribution.

## Sampling Pseudocode

```python
def sample_training_episode():
    # Root node (random)
    stack = random.uniform(10, 200)
    pot = random.uniform(2, min(100, stack * 2))
    board = deal_random_cards(3)  # Flop

    # Ranges (uniform or random)
    oop_range = [1.0] * 1326  # or random weights
    ip_range = [1.0] * 1326

    # Deal hands (blocked by board)
    available = get_unblocked_hands(board)
    oop_hand = sample_hand(available, oop_range)
    ip_hand = sample_hand(available - oop_hand, ip_range)

    # Create state
    state = PostflopState(
        stack=stack,
        pot=pot,
        board=board,
        oop_range=oop_range,
        ip_range=ip_range,
        oop_hand=oop_hand,
        ip_hand=ip_hand
    )

    return state
    # Deep PDCFR+ takes over: traverse, sample actions, collect advantages, learn
```

## What The Network Learns

Through self-play on random scenarios, the network learns patterns:

- Low SPR → More all-in decisions
- Wet board + wide range → More checking
- Tight range + dry board → More betting
- Deep stack → Complex bet sizing
- River → Polarized strategies
- OOP → More defensive play
- IP → More aggressive play

**Key**: Network learns RELATIONSHIPS between inputs and optimal play, then generalizes to ANY combination.

## Action Space

| ID | Action | Condition |
|----|--------|-----------|
| 0 | Fold | Facing bet |
| 1 | Check | No bet to call |
| 2 | Call | Facing bet |
| 3-9 | Bet 25%-150% | No bet, stack permits |
| 10-11 | Raise | Facing bet, stack permits |
| 12 | All-in | Always available |

## Production Inference

After training, the model can solve ANY input:

```python
# User query (any values)
solution = solver.solve(
    oop_range="22+,A2s+,K9s+,Q9s+",
    ip_range="TT+,AQs+,AKo",
    board=["Ks", "Jd", "7h"],
    pot=10,
    stack=30,
    hand="AsKd"
)

# Returns
{
    "strategy": {"check": 0.3, "bet_33": 0.5, "bet_67": 0.2},
    "ev": 2.5,
    "equity": 0.65
}
```

## Summary

| Phase | What Happens |
|-------|--------------|
| Training | Algorithm randoms scenarios, explores game tree, learns patterns |
| Inference | User provides ANY input, model computes optimal solution |

**The model is a SOLVER, not a LOOKUP TABLE.**
