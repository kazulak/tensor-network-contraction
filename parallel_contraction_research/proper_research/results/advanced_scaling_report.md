# Advanced Multi-Topology 7-Size Comparative Scaling Report

This report evaluates three parallel tensor contraction backends: Pure Tree-Node Parallelism, Traditional Static Slicing (4 slices), and Advanced Active Slicing.
All measurements exclude JIT compilation through double-run warmups, executed on 4 CPU threads.

### Topology: 1D MPS Chain

| Instance Size | Pure Tree-Node (s) | Static Slicing (4 slices) (s) | Advanced Active Slicing (s) | Active Slicing Speedup |
|---|---|---|---|---|
| N=10, D=4 | 0.0000s | 0.0802s | 0.0001s | 690.63x |
| N=15, D=4 | 0.0001s | 0.0780s | 0.0002s | 491.99x |
| N=20, D=4 | 0.0000s | 0.0787s | 0.0003s | 312.15x |
| N=25, D=4 | 0.0001s | 0.0753s | 0.0002s | 317.29x |
| N=30, D=4 | 0.0001s | 0.0777s | 0.0003s | 271.17x |
| N=35, D=4 | 0.0001s | 0.0804s | 0.0002s | 413.58x |
| N=40, D=4 | 0.0001s | 0.0812s | 0.0003s | 258.01x |

### Topology: 2D PEPS Grid

| Instance Size | Pure Tree-Node (s) | Static Slicing (4 slices) (s) | Advanced Active Slicing (s) | Active Slicing Speedup |
|---|---|---|---|---|
| 3x3, D=3 | 0.0001s | 0.0724s | 0.0002s | 289.74x |
| 3x3, D=4 | 0.0001s | 0.0794s | 0.0002s | 346.19x |
| 4x3, D=4 | 0.0001s | 0.0809s | 0.0003s | 273.55x |
| 4x4, D=4 | 0.0001s | 0.0832s | 0.0003s | 327.27x |
| 5x4, D=4 | 0.0002s | 0.0784s | 0.0004s | 208.67x |
| 5x5, D=4 | 0.0002s | 0.0787s | 0.0004s | 204.88x |
| 6x5, D=4 | 0.0003s | 0.0771s | 0.0006s | 130.45x |

### Topology: 3D PEPS Grid

| Instance Size | Pure Tree-Node (s) | Static Slicing (4 slices) (s) | Advanced Active Slicing (s) | Active Slicing Speedup |
|---|---|---|---|---|
| 2x2x2, D=3 | 0.0001s | 0.0788s | 0.0002s | 481.13x |
| 2x2x2, D=4 | 0.0001s | 0.0820s | 0.0003s | 279.66x |
| 3x2x2, D=3 | 0.0001s | 0.0841s | 0.0003s | 314.58x |
| 3x3x2, D=3 | 0.0005s | 0.0886s | 0.0003s | 258.14x |
| 3x3x2, D=4 | 0.0051s | 0.0921s | 0.0051s | 18.02x |
| 3x3x3, D=3 | 0.0772s | 0.1712s | 0.1079s | 1.59x |
| 3x3x3, D=4 | 3.8241s | 0.5005s | 3.1642s | 0.16x |

### Topology: Random Regular

| Instance Size | Pure Tree-Node (s) | Static Slicing (4 slices) (s) | Advanced Active Slicing (s) | Active Slicing Speedup |
|---|---|---|---|---|
| N=12, D=4 | 0.0001s | 0.0867s | 0.0002s | 347.95x |
| N=16, D=4 | 0.0001s | 0.0791s | 0.0002s | 408.20x |
| N=20, D=4 | 0.0001s | 0.0813s | 0.0003s | 254.85x |
| N=24, D=4 | 0.0001s | 0.0656s | 0.0003s | 238.48x |
| N=28, D=4 | 0.0002s | 0.0773s | 0.0003s | 226.88x |
| N=32, D=4 | 0.0002s | 0.0779s | 0.0004s | 193.78x |
| N=36, D=4 | 0.0003s | 0.0763s | 0.0004s | 178.91x |

### Topology: Binary Tree

| Instance Size | Pure Tree-Node (s) | Static Slicing (4 slices) (s) | Advanced Active Slicing (s) | Active Slicing Speedup |
|---|---|---|---|---|
| depth=3, D=4 | 0.0000s | 0.0781s | 0.0001s | 606.50x |
| depth=3, D=6 | 0.0000s | 0.0804s | 0.0002s | 422.89x |
| depth=4, D=4 | 0.0000s | 0.0804s | 0.0002s | 354.23x |
| depth=4, D=6 | 0.0001s | 0.0668s | 0.0002s | 351.00x |
| depth=5, D=4 | 0.0001s | 0.0800s | 0.0002s | 387.56x |
| depth=5, D=5 | 0.0001s | 0.0800s | 0.0002s | 390.79x |
| depth=6, D=4 | 0.0001s | 0.0796s | 0.0003s | 252.57x |

