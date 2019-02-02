# DRL_Collaboration_and_Competition

### Introduction
For this project, the Tennis environement is used. There are two agents playing tennis with the aim of keeping the ball in the game as long as possible.
This is a project to demonstrate the application of deep reinforcement learning. Specifically the DDPG algorithm applied to scenarios with multiple agents.

The training was executed in the iPython notebook. The agent is defined in file maddpg_agent.py and the aritficial neural networks used to make the agent learn is in file model.py.

The environment in which both agents play can be descriped the following way:
* state space: There are 24 dimensions of local state, i.e. each agent has its individual state information
* action space: The action space per agent is continuous and consists of 2 dimensions corresponding to the vertical and horizontal position of the tennis racket. 
* reward: A reward of +0.1 is provided to the agent hitting the ball over the net. A reward of -0.01 is provided to the agent who hits the ball out of bounds or if the agent lets the ball hit the ground. 

The environment is considered solved, when the average over 100 episodes of the maximum score between the agents is at least 0.5.

### Getting Started

1. Download the Tennis environment from one of the links below depending on your operating system.
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

2. Unzip file in this repository's root folder.

### Instructions

Setup a python environment as described [here](https://github.com/udacity/deep-reinforcement-learning#dependencies).

Then train the agents by following the instructions in `Tennis.ipynb`. 
