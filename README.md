

# Project 2: Continuous Control

### Introduction

For this project, we train an agent for solve the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment from Unity. 

![Reacher Environment](./img/reacher_random_1x.gif)
*Reacher Environment with random actions*

In the environment, we have a double-jointed arm that can move to target locations. We obtain a reward of +1 when the hand is at goal location. Thus, the goal of the agent is to maintain its hand position at the goal location for as many time as possible. 

The observation state has 33 variables that contain position, rotation, velocity and angular velocities of the arm. Given this information, the agent must select the best action to take. In this environment the agent have an action space encoded in a vector of 4 numbers, corresponding to the torque applicable to the two joints. Also, the numbers in this action vector must be a number between -1 and 1. 

The task is episodic, and in order to solve the environment, the agent must get and average score of +30 over 100 consecutive episodes.

### Getting Started
The versions we solve are different from the official in the Unity ML-Agents site. 
There are two versions of this environment, one with one agent, and one with 20. In this repository we solve the environment with one agent.

1. Download the environment from one of the links below. You only need select the environment that matches your operating system [1]:
**1 agent**
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

**20 agent**
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

[1]: In the repository we have the Mac OSX versions of the environment

2. Place the file in the repository `p2_continuous-control/` folder, and unzip the file

### Instructions
Follow the instructions in `Continuous_control.ipynb`to get started with training your own agent.

We have some files and folders to consider:
- `model.py` - the definition of the neural networks we use for the agent (we use the [DDPG Algorithm](https://arxiv.org/abs/1509.02971) with some hyperparameters adjustments)
- `ddpg_agent.py` - definition of the agent and support classes (replay buffer and Ornstein-Uhlenbeck process)
- `train_agent.py` - script for training the agent 
- `Continuous_Control.ipynb` - Here we make experiments and train the agent
- `working_weights` - folder containig trained weights for the model

#### Continuous_Control.ipynb

The notebook where we train the agents (the script is almost the same)
- We import the needed modules 
- We define the ddpg function where:
    - We initialize the environment
    ```python
    env = UnityEnvironment(file_name=ENV_FILE)
    ```
    - We initialize the agent
    ```python
    agent = Agent(state_size=states.shape[1], action_size=action_size, random_seed=2)
    ```
    
    - We train the agent for *i* episodes
    ```python
    for i_episode in range(1, n_episodes+1):
    ```
- We train the agent calling the function and observe the results

#### model.py
In this file we define the `Actor` and `Critic` classes that contain the Neural Network definition of each one.

#### ddpg_agent.py
We define the `Agent` class and the methods for interacting (`act`, `step`) and training (`learn`, `soft_update`). We also define the `ReplayBuffer` class, needed by the agent on the training of the algorithms coded and `OUNoise` class needed for adding noise with Ornstein-Uhlenbeck process on the `act` method. 

#### working_weights folder
The folder contains two files with working (solved) weights for the Actor and Critic networks of the agent.


    