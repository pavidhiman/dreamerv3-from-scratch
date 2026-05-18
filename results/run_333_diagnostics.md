# Run: 333.8 Reward — Diagnostics & Progress

## Training Summary
- **Best reward**: 333.8 (epoch 11 of quick test)
- **Final avg reward**: 280.4 (after 100 full training epochs)
- **Total training time**: 6505s (~1.8 hours)
- **WM loss**: 3.78 → 1.43
- **Rollbacks**: 11 triggered, never exceeded quick test best
- **GPU**: Tesla T4 (Colab)

## Quick Test (20 epochs)
| Epoch | Phase | WM Loss | Reward | Best |
|-------|-------|---------|--------|------|
| 0 | WM only | 3.7809 | 158.9 | 194.7 |
| 9 | WM only | 1.8708 | 189.4 | 194.7 |
| 10 | WM+Actor | 1.8200 | 217.8 | 217.8 |
| 11 | WM+Actor | 1.8433 | 333.8 | 333.8 |
| 17 | WM+Actor | 1.8323 | 107.0 | 333.8 |
| 18 | WM+Actor | 1.8058 | 215.4 | 333.8 |

## Full Training Pattern
- Every cycle: rollback → spike ~220-265 → monotonic decay over 6-9 epochs → rollback
- Actor never found a policy better than the quick test's epoch 11 result
- WM improved steadily (1.79 → 1.43) but actor couldn't exploit it

## Option C: Imagination vs Reality
- **Branch point**: step 30, root pos = (0.07, 0.06, 1.25)
- **Real trajectory**: 23 frames then humanoid fell
- **Imagined trajectory**: 23 frames generated

## Divergence Analysis (over 23 steps)
| Body Part | Avg MAE | Final Step MAE |
|-----------|---------|----------------|
| Root (height/orient) | 0.220 | 0.382 |
| Right Leg | 0.335 | 0.473 |
| Left Leg | 0.469 | 0.721 |
| Right Arm | 0.338 | 0.677 |
| Left Arm | 0.578 | 0.927 |
| Root Velocity | 0.869 | 1.065 |

## Interpretation
- World model predicts root position best, velocity worst
- Velocity diverges fastest because it changes abruptly at contact/fall events
- Real humanoid fell after 23 steps; model likely predicted continued walking
- This optimism gap is why the actor fails — it trains against an overly stable simulator
- Asymmetry between left/right limbs (Left Leg: 0.469 vs Right Leg: 0.335) suggests the fall direction matters

## Video
- 55 frames total, reward 283.1
- Humanoid takes a few steps then falls (~2 seconds of walking)
