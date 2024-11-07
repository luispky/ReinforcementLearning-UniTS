# Reinforcement Learning Project at the University of Trieste, 2023-2024

## Author: Luis Fernando Palacios Flores (MAT. SM3800038)
## Master's degree: Data Science and Artificial Intelligence (SM38)

# Short Description

This project implements the Soft Actor-Critic and Twin Delayed Deep Deterministic Policy Gradient algorithms to tackle continuous actions on some [Gymnasium](https://gymnasium.farama.org/#) environments.

# Results

Here are some of the results.

Bipedal Walker (TD3):

[![Bipedal Walker](https://github.com/luispky/ReinforcementLearning-UniTS/raw/master/TD3/results/BipedalWalker-v3/01_best_policy.gif)](https://gymnasium.farama.org/environments/classic_control/pendulum/0)

Humanoid (SAC):

![Humanoid](https://github.com/luispky/ReinforcementLearning-UniTS/blob/master/SAC/results/Humanoid-v4/01_last_policy.gif)

# Project Structure

```bash
├── env.yaml
├── SAC
│   ├── agent.py
│   └── configs
├── scripts
│   ├── environments_overview.ipynb
│   ├── main.py
│   └── results.ipynb
├── src
│   ├── environment.py
│   ├── networks.py
│   └── utils.py
└── TD3
    ├── agent.py
    └── configs
```
* The `env.yaml` file allows to create a new conda environment with the same packages utilized in this project, paramount to execute this code. To create the environment execute the command:

```linux
conda env create -f env.yaml
```

* The SAC and TD3 directories contain the implementation of the algorithms (`agent.py`) and the parameters and hyperparameters utilized are setupped in the `config` subdirectory for each environment.
* The `main.py` contains the code to run the `SAC` and `TD3` algorithms. 

# How to execute the code?

The code can be executed from the parent directory or the `scripts` directory. To execute from the parent directory the following command can be used:

```python
python scripts/main.py -env <ENVIRONMENT_NAME> -alg <ALGORITHM>
```

The options for `<ENVIRONMENT_NAME>` are:
* [pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/0)
* [walker](https://gymnasium.farama.org/environments/box2d/bipedal_walker/)
* [halfcheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/)
* [humanoid](https://gymnasium.farama.org/environments/mujoco/humanoid/)
* [ant](https://gymnasium.farama.org/environments/mujoco/ant/)
* [walker2d](https://gymnasium.farama.org/environments/mujoco/walker2d/)
* [car](https://gymnasium.farama.org/environments/box2d/car_racing/) (doesn't work well)

The options for `<ALGORITHM>`, of course, are:
* sac
* td3

In the `scripts` subdirectory the execution is as follows:

```python
python main.py -env <ENVIRONMENT_NAME> -alg <ALGORITHM>
```

Three subdirectories will be created after executing this command:
* `checkpoints` to save the models and training data
* `logs` to keep track of the hyperparameters utilized
* `results` to store the final results

Inside these directories, specific subdirectories are created for each environment.

# Link to the slides

https://docs.google.com/presentation/d/16EGlFeVgT5UstF_6QOyH6OSu9X48F7HwPjW9v2mAvzM/edit?usp=sharing
