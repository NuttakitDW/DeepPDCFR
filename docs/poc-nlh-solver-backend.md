# POC: NLH Solver Backend

Minimal backend to count combos from ranges for OOP vs IP.

## Scope

### In Scope
- Input ranges for OOP and IP
- Input board cards
- Calculate and return combo counts

### Out of Scope
- Model inference
- Action frequencies
- Equity/EV calculations

## Input

```python
@dataclass
class QueryInput:
    board: list[str]          # ["6h", "9d", "Td"]
    oop_range: list[str]      # ["AA", "KK", "AKs", "AKo", ...]
    ip_range: list[str]       # ["AA", "KK", "QQ", ...]
```

## Output

```python
@dataclass
class ComboResult:
    oop_combos: int           # Total combos for OOP range
    ip_combos: int            # Total combos for IP range
```

## Combo Calculation

```
Base combos (no blockers):
- Pair:    6 combos (AA, KK, etc.)
- Suited:  4 combos (AKs, AQs, etc.)
- Offsuit: 12 combos (AKo, AQo, etc.)

With board blockers:
- Each board card removes combos containing that card
```

## Example

```
Input:
  board = ["Ah", "Kd", "7c"]
  oop_range = ["AA", "KK", "AKs", "AKo"]
  ip_range = ["QQ", "JJ", "AQs"]

Calculation:
  OOP:
    AA: 6 -> 3 (Ah blocks 3 combos)
    KK: 6 -> 3 (Kd blocks 3 combos)
    AKs: 4 -> 2 (Ah, Kd block 2 combos)
    AKo: 12 -> 6 (Ah, Kd block 6 combos)
    Total: 14 combos

  IP:
    QQ: 6 -> 6 (no blockers)
    JJ: 6 -> 6 (no blockers)
    AQs: 4 -> 3 (Ah blocks 1 combo)
    Total: 15 combos

Output:
  oop_combos = 14
  ip_combos = 15
```

## Implementation

```
scripts/
└── poc_solver.py
```

```python
def count_combos(range_hands: list[str], board: list[str]) -> int:
    """Count available combos for a range given board blockers."""
    total = 0
    blocked = set(board)

    for hand in range_hands:
        total += get_unblocked_combos(hand, blocked)

    return total

def get_combos(query: QueryInput) -> ComboResult:
    return ComboResult(
        oop_combos=count_combos(query.oop_range, query.board),
        ip_combos=count_combos(query.ip_range, query.board)
    )
```

## Success Criteria

1. Input board + OOP range + IP range
2. Return accurate combo counts accounting for blockers
