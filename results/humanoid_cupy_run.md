# Humanoid-v4 Training Results — CuPy GPU Accelerated

## Configuration
- Environment: Humanoid-v4 (obs=376, actions=17)
- Backend: CuPy (full GPU forward+backward)
- Hidden size: 512, Stoch: 32x32 = 1024
- State size: 1536
- WM LR: 1e-4, Actor/Critic LR: 3e-5
- Batch size: 32, Seq len: 16
- WM warmup: 25 epochs, Actor update ratio: 1:5
- Imagination horizon: 4, detached between steps
- Action regularization: 0.1 * |action|^2

## Key Results
- **Random baseline: 115.3**
- **Best reward: 230.5** (epoch 72) — 2x improvement over random
- **Final avg reward: 214.3**
- **Total time: 4401s (~73 minutes)**
- Epoch time: ~54s (3x faster than NumPy version's 155s)

## Training Log

| Epoch | Phase | WM Loss | Reward | Buffer |
|-------|-------|---------|--------|--------|
| 0 | WM only | 2.2148 | 101.8 | 10000 |
| 8 | WM only | 1.2097 | 121.6 | 14000 |
| 24 | WM only | 1.1231 | 100.9 | 22000 |
| 25 | WM+Actor | 1.1209 | 137.1 | 22500 |
| 27 | WM+Actor | 1.1175 | 199.6 | 23500 |
| 28 | WM+Actor | 1.1146 | 201.7 | 24000 |
| 67 | WM+Actor | 1.0837 | 207.8 | 43500 |
| 71 | WM+Actor | 1.0802 | 215.4 | 45500 |
| 72 | WM+Actor | 1.0811 | **230.5** | 46000 |
| 79 | WM+Actor | 1.0766 | 203.8 | 49500 |

## Analysis
- Actor training DID NOT crash reward (previous bug: 155 → 71)
- Reward jumped from ~100 to ~200 within 3 epochs of actor starting
- Peaked at 230.5 — humanoid staying upright ~45+ timesteps
- WM loss decreased steadily: 2.21 → 1.08
- CuPy provided ~3x speedup over NumPy (54s vs 155s per epoch)
