# Reinforcement Learning for Poker Solver

## Overview

Deep PDCFR+ uses **Self-Play Reinforcement Learning** with **Regret Minimization** to learn optimal poker strategy without human-labeled data.

## Types of Machine Learning

| Type | Has Labels? | Has Feedback? | Example |
|------|-------------|---------------|---------|
| Supervised | ✅ Yes | From labels | Image classification |
| Unsupervised | ❌ No | ❌ No | Clustering |
| **Reinforcement** | ❌ No | ✅ From environment | **Game AI, Poker** |

## Reinforcement Learning Basics

```
┌─────────────────────────────────────────────────────────────────┐
│                    STANDARD RL FRAMEWORK                        │
│                                                                 │
│                         ┌───────────┐                           │
│                         │   Agent   │                           │
│                         │ (learner) │                           │
│                         └─────┬─────┘                           │
│                               │                                 │
│                    action     │     state, reward               │
│                         ┌─────┴─────┐                           │
│                         ▼           ▲                           │
│                    ┌─────────────────────┐                      │
│                    │    Environment      │                      │
│                    │    (poker game)     │                      │
│                    └─────────────────────┘                      │
│                                                                 │
│  Agent:        The poker player (neural network)                │
│  Environment:  The poker game rules                             │
│  State:        Board, pot, stack, hand, ranges                  │
│  Action:       Fold, check, call, bet, raise                    │
│  Reward:       Chips won/lost at showdown                       │
└─────────────────────────────────────────────────────────────────┘
```

## Why RL for Poker?

```
┌─────────────────────────────────────────────────────────────────┐
│ POKER CHALLENGES                                                │
│                                                                 │
│ 1. No "correct" answer to label                                 │
│    - GTO strategy is unknown, must be discovered                │
│    - Can't use supervised learning                              │
│                                                                 │
│ 2. Imperfect information                                        │
│    - Don't know opponent's cards                                │
│    - Must reason about probabilities (ranges)                   │
│                                                                 │
│ 3. Opponent adapts                                              │
│    - Best strategy depends on opponent's strategy               │
│    - Need game-theoretic equilibrium (Nash)                     │
│                                                                 │
│ SOLUTION: Self-play RL with regret minimization                 │
└─────────────────────────────────────────────────────────────────┘
```

## Self-Play Learning

```
┌─────────────────────────────────────────────────────────────────┐
│                       SELF-PLAY                                 │
│                                                                 │
│              ┌─────────────┐                                    │
│              │   Agent     │                                    │
│              │  (Player 1) │                                    │
│              └──────┬──────┘                                    │
│                     │                                           │
│                     │  plays against                            │
│                     ▼                                           │
│              ┌─────────────┐                                    │
│              │   Agent     │                                    │
│              │  (Player 2) │  ← Same network!                   │
│              └──────┬──────┘                                    │
│                     │                                           │
│                     ▼                                           │
│              Both players improve together                      │
│              Converges to Nash Equilibrium                      │
│                                                                 │
│  Used by: AlphaGo, Libratus, Pluribus, Deep PDCFR+              │
└─────────────────────────────────────────────────────────────────┘
```

## Regret Minimization (CFR)

The core algorithm behind Deep PDCFR+.

### What is Regret?

```
Regret = "How much better could I have done?"

Example:
  - I chose to CHECK, won 0 chips
  - If I had BET, I would have won 10 chips
  - Regret(CHECK) = 0 - 5 = -5 (bad choice)
  - Regret(BET) = 10 - 5 = +5 (good choice)

Over many iterations:
  - Actions with positive cumulative regret → play more
  - Actions with negative cumulative regret → play less
```

### Counterfactual Regret

```
┌─────────────────────────────────────────────────────────────────┐
│ COUNTERFACTUAL = "What if I had played differently?"            │
│                                                                 │
│ Standard Regret:                                                │
│   Compare actual outcome vs alternative                         │
│                                                                 │
│ Counterfactual Regret (CFR):                                    │
│   "If I ALWAYS reached this decision point,                     │
│    what would be the regret of each action?"                    │
│                                                                 │
│ This allows learning even for unlikely situations!              │
│                                                                 │
│ Formula:                                                        │
│   regret(I, a) = value(I, a) - Σ strategy(a') × value(I, a')    │
│                                                                 │
│   I = information state (what player knows)                     │
│   a = action being evaluated                                    │
│   value = expected chips from this action                       │
└─────────────────────────────────────────────────────────────────┘
```

### Regret Matching

```
Convert regrets to strategy (action probabilities):

  If all regrets ≤ 0:
    Play uniformly (all actions equal probability)

  Else:
    strategy(a) = max(regret(a), 0) / Σ max(regret(a'), 0)

Example:
  Regrets: [CHECK: -5, BET_SMALL: +10, BET_BIG: +5]

  Positive regrets: [0, 10, 5]
  Sum = 15

  Strategy: [CHECK: 0%, BET_SMALL: 67%, BET_BIG: 33%]
```

## Deep PDCFR+ Components

### 1. Cumulative Advantage Network

```
┌─────────────────────────────────────────────────────────────────┐
│ PREDICTS CUMULATIVE ADVANTAGES (REGRETS)                        │
│                                                                 │
│   Input: Information state tensor (202 dims)                    │
│            - Stack, pot, board, ranges, hand, etc.              │
│                                                                 │
│   Output: Advantage for each action                             │
│            - [fold: -10, check: +2, bet_33: +5, bet_67: +3]     │
│                                                                 │
│   Training: Bootstrapping from previous iteration               │
│            new_regret = old_regret + sampled_advantage          │
│                                                                 │
│   The network LEARNS to predict regrets, not memorize them!     │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Average Policy Network

```
┌─────────────────────────────────────────────────────────────────┐
│ PREDICTS AVERAGE STRATEGY OVER ALL ITERATIONS                   │
│                                                                 │
│   Input: Information state tensor (202 dims)                    │
│                                                                 │
│   Output: Action probabilities                                  │
│            - [fold: 0%, check: 30%, bet_33: 50%, bet_67: 20%]   │
│                                                                 │
│   This is the final Nash Equilibrium strategy!                  │
│   (Average of all strategies during training)                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Value Network (Baseline)

```
┌─────────────────────────────────────────────────────────────────┐
│ REDUCES VARIANCE IN TRAINING                                    │
│                                                                 │
│   Input: History state (full game state)                        │
│                                                                 │
│   Output: Expected value of each action                         │
│                                                                 │
│   Used as baseline to reduce noise in advantage estimates       │
│   (Variance Reduction technique from DREAM paper)               │
└─────────────────────────────────────────────────────────────────┘
```

## Training Loop

```
┌─────────────────────────────────────────────────────────────────┐
│ DEEP PDCFR+ TRAINING ITERATION                                  │
│                                                                 │
│ for iteration in 1..T:                                          │
│     for player in [OOP, IP]:                                    │
│                                                                 │
│         # 1. Sample episodes through self-play                  │
│         for k in 1..K:                                          │
│             episode = play_game_with_current_strategy()         │
│             collect_advantages(episode)                         │
│                                                                 │
│         # 2. Train cumulative advantage network                 │
│         #    (bootstrapping from previous iteration)            │
│         loss = (old_network + new_advantages - new_network)²    │
│         update_advantage_network(loss)                          │
│                                                                 │
│         # 3. Train value baseline (variance reduction)          │
│         update_value_network()                                  │
│                                                                 │
│     # 4. Train average policy network                           │
│     update_policy_network()                                     │
│                                                                 │
│     # 5. Evaluate (optional)                                    │
│     if iteration % eval_freq == 0:                              │
│         compute_exploitability()                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Convergence to Nash Equilibrium

```
┌─────────────────────────────────────────────────────────────────┐
│ WHY DOES THIS CONVERGE TO OPTIMAL PLAY?                         │
│                                                                 │
│ Iteration 1:    Random strategy, high regrets                   │
│                 "I'm making many mistakes"                      │
│                                                                 │
│ Iteration 10:   Strategy improves, regrets decrease             │
│                 "I'm learning from my mistakes"                 │
│                                                                 │
│ Iteration 100:  Strategy near optimal, low regrets              │
│                 "I rarely regret my decisions"                  │
│                                                                 │
│ Iteration 1000: Nash Equilibrium (theoretically)                │
│                 "No single action I regret on average"          │
│                                                                 │
│                                                                 │
│ THEOREM: Average strategy converges to Nash Equilibrium         │
│          in two-player zero-sum games.                          │
│                                                                 │
│ Nash Equilibrium = Unexploitable strategy (GTO)                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Concepts Summary

| Concept | Description |
|---------|-------------|
| **Agent** | Neural network that learns to play poker |
| **Environment** | Poker game rules and mechanics |
| **State** | Board, pot, stack, hand, ranges (202 dims) |
| **Action** | Fold, check, call, bet sizes, raise, all-in |
| **Reward** | Chips won/lost (counterfactual) |
| **Regret** | "How much better could I have done?" |
| **CFR** | Counterfactual Regret Minimization |
| **Self-Play** | Agent plays against itself |
| **Nash Equilibrium** | Unexploitable strategy (GTO) |

## Deep PDCFR+ vs Other RL

| Aspect | Standard RL (DQN, PPO) | Deep PDCFR+ |
|--------|------------------------|-------------|
| Goal | Maximize reward | Minimize regret |
| Opponent | Fixed or environment | Self (adapting) |
| Convergence | To best response | To Nash Equilibrium |
| Game type | Single agent | Two-player zero-sum |
| Information | Usually perfect | Imperfect (hidden cards) |

## Why This Works for Poker

```
┌─────────────────────────────────────────────────────────────────┐
│ POKER REQUIREMENTS          │ DEEP PDCFR+ SOLUTION              │
│─────────────────────────────│───────────────────────────────────│
│ Handle imperfect info       │ Information state abstraction     │
│ Handle opponent adaptation  │ Self-play convergence             │
│ Find unexploitable strategy │ Regret minimization → Nash        │
│ Generalize to new spots     │ Neural network function approx    │
│ Scale to large games        │ Sampling + neural networks        │
└─────────────────────────────────────────────────────────────────┘
```

## References

- **CFR**: Zinkevich et al., 2007 - "Regret Minimization in Games with Incomplete Information"
- **CFR+**: Tammelin, 2014 - "Solving Large Imperfect Information Games Using CFR+"
- **DeepStack**: Moravčík et al., 2017 - "DeepStack: Expert-level AI in Heads-Up No-Limit Poker"
- **Libratus**: Brown & Sandholm, 2017 - "Superhuman AI for Heads-up No-limit Poker"
- **Deep CFR**: Brown et al., 2019 - "Deep Counterfactual Regret Minimization"
- **DREAM**: Steinberger et al., 2020 - "DREAM: Deep Regret Minimization with Advantage Baselines"
- **Deep PDCFR+**: Xu et al., 2025 - "Deep Predictive Discounted CFR" (this repo)
