# Quadruped Gait Gym

A reinforcement learning research framework for training and evaluating locomotion policies in a simulated quadruped robotic environment.

This project was developed as part of the Master's thesis:

**Evaluating Reinforcement Learning Algorithms for Training Movement in Simulated Robotic Quadrupeds**
*Tyler Swenson, MS Engineering, University of Wisconsin–Milwaukee (2026)*

---

## Overview

Quadruped locomotion presents a challenging control problem involving nonlinear dynamics, coordinated joint control, stability constraints, and multi-objective optimization. This repository provides a custom Gymnasium-based simulation environment inspired by the **Petoi Bittle** robotic dog and serves as a testbed for evaluating modern deep reinforcement learning algorithms.

The framework supports both **continuous** and **multi-discrete** control spaces, allowing direct comparison of different policy architectures and learning paradigms while using a shared environment and reward structure.

### Research Goals

* Train stable forward locomotion policies for a simulated quadruped
* Compare performance across multiple reinforcement learning algorithms
* Investigate the impact of reward engineering on gait development
* Evaluate neural network architecture scaling
* Optimize training configurations using automated hyperparameter search
* Explore considerations for future sim-to-real transfer

---

## Features

### Custom Gymnasium Environment

* Simulated quadruped robot inspired by the Petoi Bittle platform
* 8-joint locomotion model
* Continuous and discrete action spaces
* Observation history support
* Contact sensing
* Phase-based gait signals
* Reward shaping for locomotion and stability

### Reinforcement Learning Algorithms

Implemented and evaluated algorithms include:

* PPO (Proximal Policy Optimization)
* A2C (Advantage Actor-Critic)
* SAC (Soft Actor-Critic)
* DDPG (Deep Deterministic Policy Gradient)

### Automated Optimization

The project includes Optuna-based optimization workflows for:

* Reward function weights
* Environment parameters
* Policy hyperparameters
* Network architectures

### Evaluation Tools

* Training reward visualization
* Multi-objective performance analysis
* Pareto optimization studies
* Comparative algorithm benchmarking

---

## Environment Design

### State Space

Observations include:

* Joint positions
* Joint velocities
* Body orientation
* Historical observations
* Foot contact states
* Periodic phase signals

### Action Space

#### Continuous Control

Each joint receives a normalized command:

```text
[-1, 1]
```

#### Discrete Control

Joint positions are discretized into angle bins:

```text
[-90°, 90°]
```

with configurable step sizes.

---

## Reward Function

The locomotion objective is formulated as a weighted combination of rewards and penalties.

### Primary Rewards

* Forward movement efficiency
* Airtime reward for stepping behavior

### Secondary Rewards

* Bent-joint posture incentives

### Primary Penalties

* Height deviation
* Posture instability
* Excessive action changes
* Paw clearance violations

### Secondary Penalties

* Lateral velocity
* Vertical velocity
* Joint jitter
* High-frequency oscillation

Reward parameters can be optimized automatically using Optuna.

---

## Results Summary

Key findings from the accompanying research include:

* Medium-sized neural networks generally outperform larger architectures in stability and consistency.
* PPO and A2C performed particularly well in discrete-action environments.
* SAC demonstrated sensitivity to reward design and hyperparameters.
* DDPG showed limited gains from increasing network complexity.
* Catastrophic forgetting and training instability remain important considerations for deep reinforcement learning locomotion systems.

These results highlight the importance of balancing:

* Algorithm selection
* Reward engineering
* Model capacity
* Environment design

rather than relying solely on larger neural networks.

---

## Repository Structure

```text
quadruped-gait-gym/
│
├── environment/             	# Gymnasium environments
├── rewards/              		# Reward and penalty components
├── scripts/             		# RL training / optimization pipelines
├── saved/                		# Saved policies and data output
├── config/               		# Experiment configurations
└── README.md
```


---

## Installation

```bash
git clone https://github.com/Swenson-EE/quadruped-gait-gym.git

cd quadruped-gait-gym

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Training

Training can be accomplished by running the `shared/training/train.py` by running it as a module from the root directory. Other scripts for optimization and complete runs can be found in the `scripts` folder.

---

## Future Work

* Domain randomization
* Sim-to-real deployment on Petoi Bittle hardware
* Terrain adaptation
* Multi-gait locomotion (walk, trot, pace, bound)
* Energy-aware control policies
* Curriculum learning approaches

---

## Thesis

This repository accompanies the research work:

**Tyler Swenson**
*Evaluating Reinforcement Learning Algorithms for Training Movement in Simulated Robotic Quadrupeds*
University of Wisconsin–Milwaukee, 2026

The thesis investigates reinforcement learning approaches for quadruped locomotion and compares PPO, A2C, SAC, and DDPG using a custom Gymnasium simulation environment.

---

## License

This project is released under the MIT License unless otherwise specified.

---

## Citation

```bibtex
@mastersthesis{swenson2026quadruped,
  title={Evaluating Reinforcement Learning Algorithms for Training Movement in Simulated Robotic Quadrupeds},
  author={Swenson, Tyler},
  school={University of Wisconsin-Milwaukee},
  year={2026}
}
```
