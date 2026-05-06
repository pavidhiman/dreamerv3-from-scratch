# Humanoid-v4 Training Results (Colab T4 GPU)

## Configuration
- Environment: Humanoid-v4 (obs=376, actions=17)
- Hidden size: 512
- Stoch latent: 32×32 = 1024
- State size: 1536
- World model params: 8,007,034
- Actor params: 1,058,321
- Critic params: 1,050,113
- WM learning rate: 1e-4
- Actor/Critic LR: 3e-5
- Batch size: 8, Seq len: 8
- Train steps/epoch: 200
- Collect steps/epoch: 1000
- Action scale: 0.4
- GPU matmul: matmul_tiled.so (CUDA)

## Results

| Epoch | WM Loss | Reward | Buffer | Time (s) |
|-------|---------|--------|--------|-----------|
| baseline | — | 155.6 | 5000 | — |
| 0 | 1.8441 | 71.1 | 5000 | 215 |
| 1 | 1.3725 | 70.9 | 6000 | 214 |
| 2 | 1.3105 | 70.9 | 7000 | 210 |
| 3 | 1.2551 | 70.9 | 8000 | 209 |
| 4 | 1.2217 | 70.9 | 9000 | 209 |
| 5 | 1.2043 | 71.0 | 10000 | 208 |
| 6 | 1.1841 | 71.0 | 11000 | 209 |
| 7 | 1.1744 | 70.9 | 12000 | 210 |
| 8 | 1.1625 | 71.0 | 13000 | 211 |
| 9 | 1.1489 | 70.9 | 14000 | 207 |
| 10 | 1.1442 | 70.9 | 15000 | 207 |
| 11 | 1.1420 | 70.9 | 16000 | 211 |
| 12 | 1.1347 | 70.8 | 17000 | 219 |
| 13 | 1.1308 | 70.9 | 18000 | 215 |
| 14 | 1.1221 | 71.0 | 19000 | 217 |
| 15 | 1.1187 | 70.9 | 20000 | 218 |
| 16 | 1.1144 | 70.8 | 21000 | 218 |
| 17 | 1.1093 | 71.1 | 22000 | 214 |
| 18 | 1.1056 | 70.9 | 23000 | 214 |
| 19 | 1.1063 | 70.8 | 24000 | 216 |

Final avg reward: 70.9
Total time: 4566.6s (~76 minutes)

## Analysis
- World model loss decreased steadily (1.84 → 1.11) — learning physics
- Reward stuck at ~71 (dropped from 155 random baseline) — actor not yet benefiting from imagination
- WM loss still decreasing at epoch 19 — hasn't converged
- Need significantly more training for actor improvement on this complex task
