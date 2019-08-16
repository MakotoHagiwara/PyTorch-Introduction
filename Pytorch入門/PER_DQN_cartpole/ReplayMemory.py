import torch
import numpy as np

class ReplayMemory:
    def __init__(self, memory_size = 10000, batch_size = 32, obs_size = 4):
        self.index = 0
        self.memory_size = memory_size
        self.batch_size = batch_size
        #場合によってはint型にしてメモリ使用量を削減してることもある
        self.obs = np.zeros((self.memory_size, obs_size), dtype = np.float)
        self.actions = np.zeros((self.memory_size, 1), dtype = np.int)
        self.rewards = np.zeros((self.memory_size, 1), dtype = np.float)
        self.next_obs = np.zeros((self.memory_size, obs_size), dtype = np.float)
        #メモリ使用量削減のためにfloatではなくintで保存する
        self.priorities = np.zeros((self.memory_size), dtype = np.float)
        self.terminates = np.zeros((self.memory_size, 1), dtype = np.int)
        
    def add(self, obs, action, reward, next_obs, priority, terminate):
        self.obs[self.index % self.memory_size] = obs
        self.actions[self.index % self.memory_size] = action
        #報酬と終端に関してはサイズ1の要素を格納している
        self.rewards[self.index % self.memory_size][0] = reward
        self.next_obs[self.index % self.memory_size] = next_obs
        self.priorities[self.index % self.memory_size] = priority
        self.terminates[self.index % self.memory_size][0] = terminate
        self.index += 1
        
    def sample(self):
        index = min(self.memory_size, self.index)
        w = np.array(self.priorities[ : index]) / np.sum(self.priorities[ : index])
        size_indices = np.arange(index)
        indices = np.random.choice(a = size_indices, 
                                   size = self.batch_size, 
                                   p = w.squeeze())
        batch = dict()
        batch['obs'] = torch.Tensor(self.obs[indices])
        #intだとNNで読み込めないのでラベルやtensorのindexとして使えないのでlongを使う必要がある
        batch['actions'] = torch.LongTensor(self.actions[indices])
        batch['rewards'] = torch.Tensor(self.rewards[indices])
        batch['next_obs'] = torch.Tensor(self.next_obs[indices])
        batch['terminates'] = torch.Tensor(self.terminates[indices])
        return batch, indices
    
    def update_priority(self, indices, priorities):
        self.priorities[indices] = priorities