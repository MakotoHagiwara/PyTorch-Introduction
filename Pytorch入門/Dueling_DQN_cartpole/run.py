import torch
import torch.nn as nn
from torch import optim
import numpy as np

import gym
import random

from model import DuelingQFunc
from ReplayMemory import ReplayMemory

lr = 1e-3
gamma = 0.95
epsilon = 0.3
batch_size = 32
initial_exploration = 500

qf = DuelingQFunc()
target_qf = DuelingQFunc()
#model.state_dict():モデルの学習パラメータをとってきている
target_qf.load_state_dict(qf.state_dict())

optimizer = optim.Adam(qf.parameters(), lr = lr)

criterion = nn.MSELoss()

memory = ReplayMemory()

env = gym.make('CartPole-v0')
obs_size =env.observation_space.shape[0]
action_size = env.action_space.n

total_step = 0

for episode in range(200):
    done = False
    obs = env.reset()
    sum_reward = 0
    step = 0
    
    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = qf.select_action(obs)
        epsilon -= 1e-4
        if epsilon < 0:
            epsilon = 0
            
        next_obs, reward, done, _ = env.step(action)
        
        terminal = 0
        reward = 0
        if done:
            terminal = 1
            if not step >= 195:
                reward = -1
        sum_reward += reward
        
        memory.add(obs, action, reward, next_obs, terminal)
        obs = next_obs.copy()
        
        step += 1
        total_step += 1
        if total_step < initial_exploration:
            continue
            
        batch = memory.sample()
        
        q_value = qf(batch['obs']).gather(1, batch['actions'])
        
        with torch.no_grad():
            next_q_value = target_qf(batch['next_obs']).max(dim = 1, keepdim = True)[0]
            target_q_value = batch['rewards'] + gamma * next_q_value * (1 - batch['terminates'])
            
        loss = criterion(q_value, target_q_value)
        
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        
        if total_step % 10 == 0:
            #targetネットワークの更新
            target_qf.load_state_dict(qf.state_dict())
            
    if episode % 10 == 0:
        print('episode:',episode, 'return:', step, 'epsilon:', epsilon)