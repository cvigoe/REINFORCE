# PyTorch implementation of REINFORCE (with Baseline)
# Author: Conor Igoe
# Date: July 8 2021

import mlflow
from mlflow.tracking import MlflowClient
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import gym
import tqdm

from buffer import Buffer
from networks import Policy, Baseline
from config import variant
from utils import evaluate, flatten_dict

def experiment(variant):
    # Create environment, policy network, optimiser and buffer
    ENV_STRING = variant['ENV_STRING']
    NUM_EPOCHS = variant['NUM_EPOCHS']
    NUM_EXPLORATION_SAMPLES_PER_EPOCH = variant['NUM_EXPLORATION_SAMPLES_PER_EPOCH']
    NUM_POLICY_GRADIENT_STEPS_PER_EPOCH = variant['NUM_POLICY_GRADIENT_STEPS_PER_EPOCH']
    NUM_BASELINE_GRADIENT_STEPS_PER_EPOCH = variant['NUM_BASELINE_GRADIENT_STEPS_PER_EPOCH']
    BATCH_SIZE = variant['BATCH_SIZE']
    POLICY_LEARNING_RATE = variant['POLICY_LEARNING_RATE']
    BASELINE_LEARNING_RATE = variant['BASELINE_LEARNING_RATE']
    HIDDEN_SIZE = variant['HIDDEN_SIZE']
    GAMMA = variant['GAMMA']
    NUM_EVALUATION_ROLLOUTS = variant['NUM_EVALUATION_ROLLOUTS']
    USE_BASELINE = variant['USE_BASELINE']
    MLFLOW_URI = variant["MLFLOW_URI"]

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

    best_evaluation_return = -1*np.inf

    for epoch in tqdm.tqdm(range(NUM_EPOCHS)):
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
        best_evaluation_return = log_performance(pi, pi_optimiser, baseline, baseline_optimiser, 
            evaluation_return, baseline_loss, epoch, 
            best_evaluation_return, MLFLOW_URI)

def log_performance(pi, pi_optimiser, baseline, baseline_optimiser, 
            evaluation_return, baseline_loss, epoch, 
            best_evaluation_return, MLFLOW_URI):
    print("Mean Return after {} Epochs: {}".format(epoch + 1, 
        evaluation_return))
    print("Baseline Loss after {} Epochs: {}".format(epoch + 1, 
        baseline_loss.detach()))
    if MLFLOW_URI != '':
        mlflow.log_metric('Evaluation Return', evaluation_return, step=epoch)
        mlflow.log_metric('Baseline Loss', baseline_loss.detach().item(), step=epoch)
        if evaluation_return > best_evaluation_return:
            torch.save(pi.state_dict(), open('data/pi_state_dict.pkl','wb'))
            torch.save(pi_optimiser.state_dict(), open('data/pi_optimiser_state_dict.pkl','wb'))
            torch.save(baseline.state_dict(), open('data/baseline_state_dict.pkl','wb'))
            torch.save(baseline_optimiser.state_dict(), open('data/baseline_optimiser_state_dict.pkl','wb'))
            mlflow.log_artifact('data/pi_state_dict.pkl')
            mlflow.log_artifact('data/pi_optimiser_state_dict.pkl')
            mlflow.log_artifact('data/baseline_state_dict.pkl')
            mlflow.log_artifact('data/baseline_optimiser_state_dict.pkl')            
            best_evaluation_return = evaluation_return    
    return best_evaluation_return

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("name_of_exp")
    parser.add_argument("name_of_run")
    parser.add_argument("mlflow_note")
    return parser.parse_args()            

if __name__ == "__main__":
    args = get_args()
    if variant["MLFLOW_URI"] == '':
        experiment(variant)
    else:
        mlflow.set_tracking_uri(variant['MLFLOW_URI'])
        mlflow.set_experiment(args.name_of_exp)
        client = MlflowClient()
        with mlflow.start_run(run_name=args.name_of_run) as run:
            mlflow.log_params(flatten_dict(variant))
            client.set_tag(run.info.run_id, "mlflow.note.content", args.mlflow_note)
            experiment(variant)
                                               