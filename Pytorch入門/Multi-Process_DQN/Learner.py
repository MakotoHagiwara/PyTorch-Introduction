import torch
import torch.nn as nn
from torch import optim
import numpy as np
import queue
import os
from time import sleep

import random
from model import DuelingQFunc
from ReplayMemory import ReplayMemory

def learner_process(path, model_path, target_model_path):
    learner = Learner(path, model_path, target_model_path)
    learner.run()
    
class Learner:
    def __init__(self, path, model_path, target_model_path):
        self.path = path
        self.model_path = model_path
        self.target_model_path = target_model_path
        self.lr = 1e-3
        self.gamma = 0.95
        self.epsilon = 0.3
        self.batch_size = 32
        self.N_STEP = 3
        self.qf = DuelingQFunc()
        self.target_qf = DuelingQFunc()
        #model.state_dict():モデルの学習パラメータをとってきている
        self.target_qf.load_state_dict(self.qf.state_dict())
        self.optimizer = optim.Adam(self.qf.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()
        self.memory = ReplayMemory()
        self.total_step = 0
        
    def run(self):
        while True:
            read_step = 0
            while True and self.total_step % 100 == 0:
                read_step += 1
                if os.path.isfile(self.path):
                    try:
                        trans_memory = torch.load(self.path)
                        os.remove(self.path)
                        self.memory.add_memory(trans_memory)
                        break
                    except:
                        sleep(np.random.random() * 2 + 2)
                elif read_step > 25:
                    break
            #一定以上の履歴が格納されていればモデルの学習を行う
            if self.memory.get_memory_size() > 100:
                batch, indices, probability_distribution = self.memory.sample()
                #各サンプルにおける状態行動の値を取ってくる
                q_value = self.qf(batch['obs']).gather(1, batch['actions'])
                #PERにおけるimportance samplingによるバイアスを打ち消すための処理
                weights = torch.tensor(np.power(probability_distribution, -1) / self.batch_size, dtype = torch.float)
        
                #サンプルごとの処理を同時に行う
                with torch.no_grad():
                    #Q-networkにおける最大値のインデックスを取ってくる
                    max_next_q_value_index = self.qf(batch['next_obs']).max(dim = 1, keepdim = True)[1]
                    #target-Q-network内の、対応する行動のインデックスにおける価値関数の値を取ってくる
                    next_q_value = self.target_qf(batch['next_obs']).gather(1, max_next_q_value_index)
                    #目的とする値の導出
                    target_q_value = batch['rewards'] + self.gamma * next_q_value * (1 - batch['terminates'])
                #PERにおけるimportance samplingによるバイアスを打ち消すための処理
                #誤差の計算
                loss = torch.mean(weights * (0.5 * (q_value - target_q_value) ** 2))
                #勾配を0にリセットする
                self.optimizer.zero_grad()
                #逆誤差伝搬を計算する
                loss.backward()
                #勾配を更新する
                self.optimizer.step()
        
                with torch.no_grad():
                    q_value = self.qf(batch['obs']).gather(1, batch['actions'])
                    #Q-networkにおける最大値のインデックスを取ってくる
                    max_next_q_value_index = self.qf(batch['next_obs']).max(dim = 1, keepdim = True)[1]
                    #target-Q-network内の、対応する行動のインデックスにおける価値関数の値を取ってくる
                    next_q_value = self.target_qf(batch['next_obs']).gather(1, max_next_q_value_index)
                    #目的とする値の導出
                    target_q_value = batch['rewards'] + self.gamma * next_q_value * (1 - batch['terminates'])
                    priorities = (abs(target_q_value - q_value)).numpy().squeeze()
                    self.memory.update_priority(indices, priorities)
        
                if self.total_step % 50 == 0:
                    #targetネットワークの更新
                    torch.save(self.qf.state_dict(), self.model_path)
                    torch.save(self.target_qf.state_dict(), self.target_model_path)
                    self.target_qf.load_state_dict(self.qf.state_dict())
                self.total_step += 1    