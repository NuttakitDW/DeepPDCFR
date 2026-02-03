# Model Validation Method

## Overview

This document describes the validation approach for Deep PDCFR+ trained models. Since computing exact exploitability is prohibitively expensive for large postflop games, we use a pre-computed validation set to measure model quality.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VALIDATION PIPELINE                          │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Pre-solve   │    │    Train     │    │   Validate   │      │
│  │  500 spots   │───►│ Deep PDCFR+  │───►│  deviation   │      │
│  │  (one-time)  │    │              │    │   < 1% ?     │      │
│  └──────────────┘    └──────────────┘    └──────┬───────┘      │
│                             ▲                    │              │
│                             │       No           │              │
│                             └────────────────────┘              │
│                                                  │ Yes          │
│                                                  ▼              │
│                                         ┌──────────────┐       │
│                                         │  Model Ready │       │
│                                         │  solver.pt   │       │
│                                         └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

## Phase 1: Validation Set Creation (One-Time)

### Scenario Selection

Select 500 representative scenarios covering:

| Dimension | Values | Count |
|-----------|--------|-------|
| Board Texture | Dry, Wet, Monotone, Paired, Connected | 5 |
| SPR | 2, 4, 6, 8, 10 | 5 |
| Street | Flop, Turn, River | 3 |
| Range Type | SRP, 3bet, 4bet | 3 |
| Position Matchup | BTNvsBB, COvsBB, SBvsBB | 3 |

Total combinations: ~500 strategically diverse scenarios

### Example Scenarios

```yaml
validation_scenarios:
  - id: 1
    board: ["Ks", "Jd", "7h"]
    street: flop
    pot: 10
    stack: 30
    spr: 3.0
    oop_range: "22+,A2s+,K9s+,Q9s+,J9s+,T8s+,97s+,87s,76s,65s,A9o+,KTo+,QTo+,JTo"
    ip_range: "TT+,AJs+,KQs,AQo+"

  - id: 2
    board: ["Ac", "Kc", "5c"]  # Monotone
    street: flop
    pot: 20
    stack: 30
    spr: 1.5
    oop_range: "..."
    ip_range: "..."
```

### Solving Process

```python
for scenario in validation_scenarios:
    # Solve with traditional DCFR+ (exact)
    solution = dcfr_plus_solve(
        scenario,
        iterations=10000,
        target_exploitability=0.001  # 0.1% pot
    )

    # Store complete solution
    store_solution(scenario.id, {
        'strategy': solution.strategy,      # Per-hand action probabilities
        'ev': solution.ev,                  # Per-hand expected values
        'equity': solution.equity,          # Per-hand equity
        'exploitability': solution.exp,     # Final exploitability
    })
```

### Storage Format

```
validation/
├── scenarios.json          # Scenario definitions
├── solutions/
│   ├── scenario_001.npz    # Strategy, EV, equity arrays
│   ├── scenario_002.npz
│   └── ...
└── metadata.json           # Solve parameters, timestamps
```

## Phase 2: Validation During Training

### Check Frequency

```python
VALIDATION_FREQUENCY = 100  # Check every 100 iterations
```

### Validation Function

```python
def validate_model(model, validation_set):
    """
    Compare model predictions against exact solutions.

    Returns:
        dict: Deviation metrics
    """
    strategy_deviations = []
    ev_deviations = []

    for scenario in validation_set:
        # Load exact solution
        exact = load_solution(scenario.id)

        # Get model prediction
        state = create_state(scenario)
        model_strategy = model.get_strategy(state)
        model_ev = model.get_ev(state)

        # Calculate deviations
        strategy_dev = compute_strategy_deviation(
            model_strategy,
            exact.strategy
        )
        ev_dev = compute_ev_deviation(
            model_ev,
            exact.ev,
            scenario.pot
        )

        strategy_deviations.append(strategy_dev)
        ev_deviations.append(ev_dev)

    return {
        'strategy': {
            'mean': np.mean(strategy_deviations),
            'max': np.max(strategy_deviations),
            'std': np.std(strategy_deviations),
            'p95': np.percentile(strategy_deviations, 95),
        },
        'ev': {
            'mean': np.mean(ev_deviations),
            'max': np.max(ev_deviations),
            'std': np.std(ev_deviations),
            'p95': np.percentile(ev_deviations, 95),
        },
        'worst_scenarios': get_worst_scenarios(
            strategy_deviations,
            ev_deviations,
            n=10
        ),
    }
```

### Deviation Metrics

#### Strategy Deviation

Mean absolute difference in action probabilities across all hands:

```python
def compute_strategy_deviation(model_strategy, exact_strategy):
    """
    model_strategy: dict[hand] -> dict[action] -> probability
    exact_strategy: dict[hand] -> dict[action] -> probability

    Returns: float (0.0 to 1.0)
    """
    deviations = []

    for hand in exact_strategy.keys():
        for action in exact_strategy[hand].keys():
            model_prob = model_strategy.get(hand, {}).get(action, 0)
            exact_prob = exact_strategy[hand][action]
            deviations.append(abs(model_prob - exact_prob))

    return np.mean(deviations)
```

#### EV Deviation

Absolute EV difference as percentage of pot:

```python
def compute_ev_deviation(model_ev, exact_ev, pot_size):
    """
    Returns: float (percentage of pot)
    """
    ev_diff = abs(model_ev - exact_ev)
    return ev_diff / pot_size * 100
```

## Phase 3: Acceptance Criteria

### Primary Metrics

| Metric | Target | Hard Limit | Description |
|--------|--------|------------|-------------|
| Avg Strategy Dev | < 2% | < 3% | Mean action probability error |
| Max Strategy Dev | < 5% | < 8% | Worst-case action error |
| Avg EV Dev | < 1% pot | < 2% pot | Mean EV error |
| Max EV Dev | < 3% pot | < 5% pot | Worst-case EV error |

### Acceptance Logic

```python
def is_model_acceptable(metrics):
    """
    Check if model meets production quality standards.
    """
    criteria = {
        'strategy_mean': metrics['strategy']['mean'] < 0.02,
        'strategy_max': metrics['strategy']['max'] < 0.05,
        'ev_mean': metrics['ev']['mean'] < 1.0,
        'ev_max': metrics['ev']['max'] < 3.0,
    }

    passed = all(criteria.values())

    return {
        'passed': passed,
        'criteria': criteria,
        'recommendation': 'SHIP IT!' if passed else 'CONTINUE TRAINING'
    }
```

### Example Validation Report

```
╔═══════════════════════════════════════════════════════════════════╗
║                    VALIDATION REPORT                              ║
║                    Iteration: 800                                 ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Scenarios Tested: 500                                            ║
║                                                                   ║
║  STRATEGY DEVIATION                                               ║
║  ┌─────────────────────────────────────────────────────────────┐  ║
║  │ Mean:   1.3%  ████████████░░░░░░░░  ✓ (target < 2%)        │  ║
║  │ Max:    4.1%  ████████████████░░░░  ✓ (target < 5%)        │  ║
║  │ P95:    2.8%  ██████████████░░░░░░  ✓                      │  ║
║  │ Std:    0.9%                                                │  ║
║  └─────────────────────────────────────────────────────────────┘  ║
║                                                                   ║
║  EV DEVIATION (% of pot)                                          ║
║  ┌─────────────────────────────────────────────────────────────┐  ║
║  │ Mean:   0.7%  ███████░░░░░░░░░░░░░  ✓ (target < 1%)        │  ║
║  │ Max:    2.1%  ██████████████░░░░░░  ✓ (target < 3%)        │  ║
║  │ P95:    1.5%  ███████████░░░░░░░░░  ✓                      │  ║
║  │ Std:    0.5%                                                │  ║
║  └─────────────────────────────────────────────────────────────┘  ║
║                                                                   ║
║  WORST SCENARIOS                                                  ║
║  ┌─────────────────────────────────────────────────────────────┐  ║
║  │ 1. Scenario 142: Monotone A-high, 3bet pot    (4.1% strat) │  ║
║  │ 2. Scenario 287: Paired board, deep stack     (3.8% strat) │  ║
║  │ 3. Scenario 051: Dry K-high, single raised    (3.2% strat) │  ║
║  └─────────────────────────────────────────────────────────────┘  ║
║                                                                   ║
║  ════════════════════════════════════════════════════════════════ ║
║  ✅ ALL CRITERIA PASSED                                           ║
║  Recommendation: Model ready for production!                      ║
╚═══════════════════════════════════════════════════════════════════╝
```

## Phase 4: Targeted Retraining (If Needed)

### Identify Weak Spots

```python
def identify_weak_spots(validation_results):
    """
    Analyze which scenario types have highest deviation.
    """
    weak_spots = []

    for scenario, deviation in validation_results['by_scenario'].items():
        if deviation['strategy'] > 0.03:  # > 3% deviation
            weak_spots.append({
                'scenario': scenario,
                'deviation': deviation,
                'features': extract_features(scenario),
            })

    # Group by common features
    patterns = group_by_features(weak_spots)

    return patterns
    # e.g., [
    #   {'pattern': 'monotone_board', 'count': 15, 'avg_dev': 3.8%},
    #   {'pattern': '3bet_pot', 'count': 12, 'avg_dev': 3.2%},
    # ]
```

### Generate Focused Training Data

```python
def generate_focused_data(weak_patterns, num_scenarios=10000):
    """
    Generate additional training scenarios for weak spots.
    """
    scenarios = []

    for pattern in weak_patterns:
        weight = pattern['avg_dev']  # Higher deviation = more samples
        num_samples = int(num_scenarios * weight / sum_weights)

        if pattern['pattern'] == 'monotone_board':
            scenarios.extend(generate_monotone_boards(num_samples))
        elif pattern['pattern'] == '3bet_pot':
            scenarios.extend(generate_3bet_pots(num_samples))
        # ... etc

    return scenarios
```

### Retraining Loop

```python
def targeted_retrain(model, weak_patterns, validation_set):
    """
    Continue training with focus on weak spots.
    """
    # Generate focused data
    focused_data = generate_focused_data(weak_patterns)

    # Mix with regular training data (70% focused, 30% random)
    training_mix = {
        'focused': 0.7,
        'random': 0.3,
    }

    # Continue training
    while True:
        # Sample with bias toward weak spots
        batch = sample_mixed_batch(focused_data, training_mix)
        model.train_step(batch)

        # Validate periodically
        if iteration % 100 == 0:
            metrics = validate_model(model, validation_set)
            if is_model_acceptable(metrics):
                break

    return model
```

## Implementation Checklist

### One-Time Setup
- [ ] Select 500 representative scenarios
- [ ] Solve all scenarios with traditional DCFR+
- [ ] Store solutions in `validation/` directory
- [ ] Verify solution quality (exploitability < 0.1%)

### Training Integration
- [ ] Implement `validate_model()` function
- [ ] Add validation check every 100 iterations
- [ ] Log validation metrics to tensorboard/wandb
- [ ] Implement early stopping on acceptance criteria

### Reporting
- [ ] Generate validation report after training
- [ ] Identify and log worst-performing scenarios
- [ ] Track validation metrics over training history

### Targeted Retraining
- [ ] Implement weak spot identification
- [ ] Implement focused data generation
- [ ] Implement mixed sampling for retraining

## Quality Levels

| Level | Avg Strategy Dev | Use Case |
|-------|------------------|----------|
| Research | < 5% | Experimentation, prototyping |
| Beta | < 2% | Internal testing, early users |
| Production | < 1% | Public release |
| Professional | < 0.5% | High-stakes, competitive |

## Notes

- Validation set should be **held out** from training data
- Re-validate with fresh scenarios periodically to check overfitting
- Consider stratified sampling to ensure all scenario types are covered
- For production, run validation on multiple random seeds

## References

- Deep Predictive DCFR+ Paper (2511.08174v1)
- Original CFR Paper (Zinkevich et al. 2007)
- DeepStack Paper (Moravčík et al. 2017)
