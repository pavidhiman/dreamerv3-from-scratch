# Pendulum-v1 Training Results — Colab GPU Run
**Date:** May 5, 2026  
**GPU:** Tesla T4 (15GB)  
**CUDA Kernel:** gpu_ops/matmul_tiled.so (tiled shared memory matmul)  
**Total time:** 9293s (~2.5 hours)

## Configuration
- hidden_size: 128
- stoch_size: 16, stoch_classes: 16
- Model params: 478,725
- Actor params: 65,921
- Critic params: 65,921
- 30 epochs, 500 train steps/epoch, 300 collect steps/epoch
- batch_size: 16, seq_len: 16, horizon: 15

## Results
```
Before training: -1545.6

epoch  0: wm_loss=1.1781, reward=-1367.2, buffer=2000
epoch  1: wm_loss=1.0284, reward=-1462.4, buffer=2300
epoch  2: wm_loss=1.0202, reward=-1565.4, buffer=2600
epoch  3: wm_loss=1.0158, reward=-1289.4, buffer=2900
epoch  4: wm_loss=1.0134, reward=-1493.4, buffer=3200
epoch  5: wm_loss=1.0123, reward=-1440.5, buffer=3500
epoch  6: wm_loss=1.0105, reward=-1409.9, buffer=3800
epoch  7: wm_loss=1.0098, reward=-1411.5, buffer=4100
epoch  8: wm_loss=1.0099, reward=-1295.3, buffer=4400
epoch  9: wm_loss=1.0093, reward=-1587.9, buffer=4700
epoch 10: wm_loss=1.0091, reward=-1581.9, buffer=5000
epoch 11: wm_loss=1.0082, reward=-1581.9, buffer=5300
epoch 12: wm_loss=1.0078, reward=-1454.0, buffer=5600
epoch 13: wm_loss=1.0098, reward=-1093.7, buffer=5900
epoch 14: wm_loss=1.0075, reward=-1090.3, buffer=6200
epoch 15: wm_loss=1.0073, reward=-1353.0, buffer=6500
epoch 16: wm_loss=1.0070, reward=-1165.4, buffer=6800
epoch 17: wm_loss=1.0066, reward=-1226.4, buffer=7100
epoch 18: wm_loss=1.0065, reward=-1277.4, buffer=7400
epoch 19: wm_loss=1.0066, reward=-1100.9, buffer=7700
epoch 20: wm_loss=1.0064, reward=-1068.1, buffer=8000
epoch 21: wm_loss=1.0063, reward=-1136.5, buffer=8300
epoch 22: wm_loss=1.0083, reward=-1297.5, buffer=8600
epoch 23: wm_loss=1.0063, reward=-1009.7, buffer=8900
epoch 24: wm_loss=1.0064, reward=-844.7, buffer=9200
epoch 25: wm_loss=1.0062, reward=-821.6, buffer=9500
epoch 26: wm_loss=1.0062, reward=-767.0, buffer=9800
epoch 27: wm_loss=1.0056, reward=-556.7, buffer=10100
epoch 28: wm_loss=1.0059, reward=-588.6, buffer=10400
epoch 29: wm_loss=1.0058, reward=-605.6, buffer=10700

Final avg reward = -538.5
```

## Summary
- Random baseline: -1545.6
- Final (10-episode avg): -538.5
- Best single epoch: -556.7 (epoch 27)
- Improvement: 65% reduction in penalty
- Pendulum reward range: -1600 (worst) to 0 (perfect)
- Breakthrough occurred around epoch 13-14, steady improvement from epoch 23 onward
