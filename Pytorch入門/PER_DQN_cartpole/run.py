import torch
import torch.nn as nn
from torch import optim
import numpy as np
import queue

import gym
import random

from model import DuelingQFunc
from ReplayMemory import ReplayMemory

lr = 1e-3
gamma = 0.95
epsilon = 0.3
batch_size = 32
initial_exploration = 500
N_STEP = 3
step_reward = 0

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
obs_queue = queue.Queue()
reward_queue = queue.Queue()
action_queue = queue.Queue()

total_step = 0
ten_step = 0

for episode in range(200):
    done = False
    obs = env.reset()
    sum_reward = 0
    step = 0
    step_reward = 0
    obs_queue = queue.Queue()
    reward_queue = queue.Queue()
    action_queue = queue.Queue()
    
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
        
        obs_queue.put(obs)
        reward_queue.put(reward)
        action_queue.put(action)
        step_reward = step_reward / gamma + reward * (gamma ** N_STEP)
        
        if step >= N_STEP - 1:
            with torch.no_grad():
                max_next_q_value_index = qf(torch.Tensor([next_obs])).max(dim = 1, keepdim = True)[1].numpy().squeeze()
                max_next_q_value = target_qf(torch.Tensor([next_obs]))[0][max_next_q_value_index].numpy()
                current_state = obs_queue.get()
                current_action = action_queue.get()
                q_value = qf(torch.Tensor([current_state]))[0][current_action].numpy()
                td_error = abs(step_reward + max_next_q_value * (gamma ** N_STEP) - q_value)
                priority = td_error
                memory.add(current_state, current_action, step_reward, next_obs, priority, terminal)
                step_reward -= reward_queue.get()
            
        if done:
            while not action_queue.empty():
                with torch.no_grad():
                    step_reward = step_reward / gamma
                    max_next_q_value_index = qf(torch.Tensor([next_obs])).max(dim = 1, keepdim = True)[1].numpy().squeeze()
                    max_next_q_value = target_qf(torch.Tensor([next_obs]))[0][max_next_q_value_index].numpy()
                    current_state = obs_queue.get()
                    current_action = action_queue.get()
                    q_value = qf(torch.Tensor([current_state]))[0][current_action].numpy()
                    td_error = abs(step_reward + max_next_q_value * (gamma ** N_STEP) - q_value)
                    priority = td_error
                    memory.add(current_state, current_action, step_reward, next_obs, priority, terminal)
                    step_reward -= reward_queue.get()
        obs = next_obs.copy()
        
        step += 1
        total_step += 1
        if total_step < initial_exploration:
            continue
            
        batch, indices, probability_distribution = memory.sample()
        
        #各サンプルにおける状態行動の値を取ってくる
        q_value = qf(batch['obs']).gather(1, batch['actions'])
        #PERにおけるimportance samplingによるバイアスを打ち消すための処理
        weight = torch.tensor(np.power(probability_distribution, -1) / batch_size, dtype = torch.float)
        q_value = q_value * weight
        
        #サンプルごとの処理を同時に行う
        with torch.no_grad():
            #Q-networkにおける最大値のインデックスを取ってくる
            max_next_q_value_index = qf(batch['next_obs']).max(dim = 1, keepdim = True)[1]
            #target-Q-network内の、対応する行動のインデックスにおける価値関数の値を取ってくる
            next_q_value = target_qf(batch['next_obs']).gather(1, max_next_q_value_index)
            #目的とする値の導出
            target_q_value = batch['rewards'] + gamma * next_q_value * (1 - batch['terminates'])
            #PERにおけるimportance samplingによるバイアスを打ち消すための処理
            target_q_value = target_q_value * weight
        
        #誤差の計算
        loss = criterion(q_value, target_q_value)
        #勾配を0にリセットする
        optimizer.zero_grad()
        #逆誤差伝搬を計算する
        loss.backward()
        #勾配を更新する
        optimizer.step()
        
        with torch.no_grad():
            q_value = qf(batch['obs']).gather(1, batch['actions'])
            #Q-networkにおける最大値のインデックスを取ってくる
            max_next_q_value_index = qf(batch['next_obs']).max(dim = 1, keepdim = True)[1]
            #target-Q-network内の、対応する行動のインデックスにおける価値関数の値を取ってくる
            next_q_value = target_qf(batch['next_obs']).gather(1, max_next_q_value_index)
            #目的とする値の導出
            target_q_value = batch['rewards'] + gamma * next_q_value * (1 - batch['terminates'])
            priorities = (abs(target_q_value - q_value)).numpy().squeeze()
            memory.update_priority(indices, priorities)
        
        if total_step % 10 == 0:
            #targetネットワークの更新
            target_qf.load_state_dict(qf.state_dict())
            
    ten_step += step
    if episode % 10 == 0:
        print('episode:',episode, 'return:', ten_step / 10.0, 'epsilon:', epsilon)
        ten_step = 0