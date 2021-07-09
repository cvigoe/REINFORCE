# PyTorch implementation of REINFORCE (with Baseline)
# Author: Conor Igoe
# Date: July 8 2021

import mlflow
import torch
import torch.nn as nn
import numpy as np
import random
import gym

from buffer import Buffer
from networks import Policy, Baseline
from config import *
from utils import *

# Create environment, policy network, optimiser and buffer
env = gym.make(ENV_STRING)
pi = Policy(env.observation_space.shape[0], 
    env.action_space.n, HIDDEN_SIZE)
baseline = Baseline(env.observation_space.shape[0], 
    HIDDEN_SIZE)
pi_optimiser = torch.optim.Adam(pi.parameters(),
    lr=POLICY_LEARNING_RATE)
baseline_optimiser = torch.optim.Adam(baseline.parameters(),
    lr=BASELINE_LEARNING_RATE)
MSELoss = torch.nn.MSELoss()
buffer = Buffer(NUM_EXPLORATION_SAMPLES_PER_EPOCH, GAMMA)

for epoch in range(NUM_EPOCHS):
    # Collect new exploration data
    with torch.no_grad():
        buffer.reset()
        while(len(buffer)) < NUM_EXPLORATION_SAMPLES_PER_EPOCH:
            state, done, path = env.reset(), False, []
            while not done:
                action = pi.act(state)
                next_state, reward, done, info = env.step(action)
                path.append([state, action, reward, next_state, done])
                state = next_state
            buffer.add_path(path)

    # Train baseline on new exploration data
    for step in range(NUM_BASELINE_GRADIENT_STEPS_PER_EPOCH):
        data = buffer.sample(BATCH_SIZE)
        states, rturns = data.state, torch.tensor(data.rturn)
        baseline_loss = MSELoss(baseline(states), rturns.unsqueeze(1))
        baseline_loss.backward()
        baseline_optimiser.step()
        baseline_optimiser.zero_grad()
    
    # Train policy on new exploration data
    for step in range(NUM_POLICY_GRADIENT_STEPS_PER_EPOCH):
        data = buffer.sample(BATCH_SIZE)
        states, actions,rturns = \
            data.state, data.action, torch.tensor(data.rturn)
        pi_loss = -1 * torch.log(
            pi(states)[torch.arange(len(states)), actions]) * \
            (rturns - baseline(states)*USE_BASELINE)
        pi_loss.mean().backward()
        pi_optimiser.step()
        pi_optimiser.zero_grad()

    # Evaluate new policy
    evaluation_return = evaluate(pi, env, NUM_EVALUATION_ROLLOUTS)
    print("Mean Return after {} / {} Epochs: {}".format(epoch + 1, 
        NUM_EPOCHS, evaluation_return))
    print("Baseline Loss after {} / {} Epochs: {}".format(epoch + 1, 
        NUM_EPOCHS, baseline_loss.detach()))

