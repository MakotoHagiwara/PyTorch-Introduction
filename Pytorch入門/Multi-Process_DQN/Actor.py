import torch
import torch.nn as nn
from torch import optim
import numpy as np
import queue

import gym
import random
from time import sleep
import os

from model import DuelingQFunc
from ReplayMemory import ReplayMemory

def actor_process(path, model_path, target_model_path):
    actor = Actor(path, model_path, target_model_path)
    actor.run()

class Actor:
    def __init__(self, path, model_path, target_model_path):
        self.path = path
        self.model_path = model_path
        self.target_model_path = target_model_path
        self.lr = 1e-3
        self.gamma = 0.95
        self.epsilon = 0.3
        self.batch_size = 32
        self.initial_exploration = 500
        self.N_STEP = 3
        self.step_reward = 0
        self.qf = DuelingQFunc()
        self.target_qf = DuelingQFunc()
        #model.state_dict():モデルの学習パラメータをとってきている
        self.target_qf.load_state_dict(self.qf.state_dict())
    
        self.optimizer = optim.Adam(self.qf.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()
        self.env = gym.make('CartPole-v0')
        self.obs_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.obs_queue = queue.Queue()
        self.reward_queue = queue.Queue()
        self.action_queue = queue.Queue()
        self.total_step = 0
        self.ten_step = 0
        self.temporal_memory = ReplayMemory()
        
    def run(self):
        for episode in range(200):
            done = False
            obs = self.env.reset()
            sum_reward = 0
            step = 0
            self.step_reward = 0
            self.obs_queue = queue.Queue()
            self.reward_queue = queue.Queue()
            self.action_queue = queue.Queue()
    
            while not done:
                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = self.qf.select_action(obs)
                self.epsilon -= 1e-4
                if self.epsilon < 0:
                    self.epsilon = 0
            
                next_obs, reward, done, _ = self.env.step(action)        
                terminal = 0
                reward = 0
                if done:
                    terminal = 1
                    if not step >= 195:
                        reward = -1
                sum_reward += reward
        
                self.obs_queue.put(obs)
                self.reward_queue.put(reward)
                self.action_queue.put(action)
                self.step_reward = self.step_reward / self.gamma + reward * (self.gamma ** self.N_STEP)
                if step >= self.N_STEP - 1:
                    with torch.no_grad():
                        max_next_q_value_index = self.qf(torch.Tensor([next_obs])).max(dim = 1, keepdim = True)[1].numpy().squeeze()
                        max_next_q_value = self.target_qf(torch.Tensor([next_obs]))[0][max_next_q_value_index].numpy()
                        current_state = self.obs_queue.get()
                        current_action = self.action_queue.get()
                        q_value = self.qf(torch.Tensor([current_state]))[0][current_action].numpy()
                        td_error = abs(self.step_reward + max_next_q_value * (self.gamma ** self.N_STEP) - q_value)
                        priority = td_error
                        self.temporal_memory.add(current_state, current_action, self.step_reward, next_obs, priority, terminal)
                        self.step_reward -= self.reward_queue.get()
                if done:
                    while not self.action_queue.empty():
                        with torch.no_grad():
                            self.step_reward = self.step_reward / self.gamma
                            max_next_q_value_index = self.qf(torch.Tensor([next_obs])).max(dim = 1, keepdim = True)[1].numpy().squeeze()
                            max_next_q_value = self.target_qf(torch.Tensor([next_obs]))[0][max_next_q_value_index].numpy()
                            current_state = self.obs_queue.get()
                            current_action = self.action_queue.get()
                            q_value = self.qf(torch.Tensor([current_state]))[0][current_action].numpy()
                            td_error = abs(self.step_reward + max_next_q_value * (self.gamma ** self.N_STEP) - q_value)
                            priority = td_error
                            self.temporal_memory.add(current_state, current_action, self.step_reward, next_obs, priority, terminal)
                            self.step_reward -= self.reward_queue.get()
                    while True:
                        try:
                            if os.path.isfile(self.path):
                                #メモリを読み込む
                                trans_memory = torch.load(self.path)
                                #メモリファイルの削除
                                os.remove(self.path)
                                #メモリに追加
                                #vstackは一番深い層の要素同士を結合する(http://ailaby.com/vstack_hstack/)
                                #vstack = concatenate(axis = 0)
                                #hstack = concatenate(axis = 1)
                                temporal_memory_size = self.temporal_memory.get_memory_size()
                                trans_memory['obs'] = np.vstack((trans_memory['obs'], self.temporal_memory.obs[ : temporal_memory_size]))
                                trans_memory['action'] = np.vstack((trans_memory['action'], self.temporal_memory.actions[ : temporal_memory_size]))
                                trans_memory['reward'] = np.vstack((trans_memory['reward'], self.temporal_memory.rewards[ : temporal_memory_size]))
                                trans_memory['next_obs'] = np.vstack((trans_memory['next_obs'], self.temporal_memory.next_obs[ : temporal_memory_size]))
                                trans_memory['priority'] = np.hstack((trans_memory['priority'], self.temporal_memory.priorities[ : temporal_memory_size]))
                                trans_memory['terminate'] = np.vstack((trans_memory['terminate'], self.temporal_memory.terminates[ : temporal_memory_size]))
                                #メモリを保存
                                torch.save(trans_memory, self.path)
                                self.temporal_memory = ReplayMemory()
                                break
                            else:
                                trans_memory = dict()
                                temporal_memory_size = self.temporal_memory.get_memory_size()
                                trans_memory['obs'] = self.temporal_memory.obs[ : temporal_memory_size]
                                trans_memory['action'] = self.temporal_memory.actions[ : temporal_memory_size]
                                trans_memory['reward'] = self.temporal_memory.rewards[ : temporal_memory_size]
                                trans_memory['next_obs'] = self.temporal_memory.next_obs[ : temporal_memory_size]
                                trans_memory['priority'] = self.temporal_memory.priorities[ : temporal_memory_size]
                                trans_memory['terminate'] = self.temporal_memory.terminates[ : temporal_memory_size]
                                torch.save(trans_memory, self.path)
                                self.temporal_memory = ReplayMemory()
                                break
                        except:
                            #他のプロセスがファイルを開いている場合は、タイミングをずらして開く
                            sleep(np.random.random() * 2 + 2)
                obs = next_obs.copy()
        
                step += 1
                self.total_step += 1
                if self.total_step < self.initial_exploration:
                    continue
            
                if self.total_step % 10 == 0:
                    #Learnerに基づいたネットワークの更新
                    while True:
                        if os.path.isfile(self.model_path):
                            try:
                                self.qf.load_state_dict(torch.load(self.model_path))
                                self.target_qf.load_state_dict(torch.load(self.target_model_path))
                                break
                            except FileNotFoundError:
                                sleep(np.random.random() * 2 + 2)
            
            self.ten_step += step
            if episode % 10 == 0:
                print('episode:',episode, 'return:', self.ten_step / 10.0, 'epsilon:', self.epsilon)
                self.ten_step = 0