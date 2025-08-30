# LIMARL: Latent-Inference Multi-Agent Reinforcement Learning

LIMARL is a framework for cooperative multi-agent reinforcement learning (MARL) under **partial observability**. It introduces **latent-state inference** to help agents coordinate effectively by learning compact representations of hidden environment dynamics. LIMARL is built on top of [PyMARL2](https://github.com/hijkzzz/pymarl2) and extends it with novel modules for latent representation learning and robust policy optimization.

---

## 🚀 Key Features

- **Latent State Representation**: Uses a **State Representation Module (SRM)** and a **Recurrent Observation-to-Latent Inference Module (ROLIM)** to infer hidden environment dynamics.
- **CTDE Framework**: Centralized Training with Decentralized Execution (CTDE).
- **Benchmark Coverage**: Evaluated on **SMAC** and **SMACv2** benchmarks (e.g., `corridor`, `MMM2`, `6h_vs_8z`).
- **Baseline Comparisons**: Includes QMIX, MA2E, QPLEX, and others.
- **Modular Implementation**: Easily extendable for new environments and MARL algorithms.

---

## 📦 Installation

Clone the repository:

```
git remote add origin https://github.com/salmakh1/LIMARL.git
cd LIMARL
```
Create a conda environment (recommended):
```
# require Anaconda 3 or Miniconda 3
conda create -n limarl python=3.8 -y
conda activate limarl

bash install_dependecies.sh

```

Install SMAC:

```bash install_sc2.sh```



# 🏃 Usage
## Training
To train LIMARL on corridor:
```
python src/main.py --config=limarl --env-config=sc2 with env_args.map_name=corridor
```

## Running Baselines

To run baselines such as QMIX, MA2E, or QPLEX (example: QMIX on MMM2):
```
python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=MMM2
```

# 📊 Results

LIMARL significantly outperforms baselines on hard exploration tasks such as corridor.

Demonstrates robust improvements in SMACv2 environments with increased partial observability.

For detailed plots, training curves, and statistical analysis (e.g., Wilcoxon signed-rank tests), please refer to our paper.


# 📂 Repository Structure

```
LIMARL/
├── src/                # Core implementation
│   ├── learners/       # Algorithm learners (LIMARL, QMIX, etc.)
│   ├── modules/        # SRM, ROLIM, and network components
│   ├── run/            # Training loop
│   └── envs/           # SMAC, SMACv2 wrappers
├── configs/            # Experiment configuration files
├── results/            # Logs and trained models
└── README.md           # This file
```

