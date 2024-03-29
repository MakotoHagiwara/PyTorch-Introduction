import torch
import torch.nn as nn

def preprocess(x):
    x = torch.tensor(x, dtype = torch.float).unsqueeze(0)
    return x

class DuelingQFunc(nn.Module):
    def __init__(self, obs_size = 4, action_size = 2):
        super(DuelingQFunc, self).__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        #dueling networkは状態以下の部分でネットワークを2分割して価値観数Vとアドバンテージ関数Aを推定する(ただしネットワークを分離しているだけ)
        self.shared_l1 = nn.Linear(in_features = obs_size, out_features = 50)
        #価値観数Vのネットワーク
        self.value_layer = nn.Linear(in_features = 50, out_features = 1)
        #アドバンテージ関数Aのネットワーク
        self.advantage_layer = nn.Linear(in_features = 50, out_features = action_size)
        self.q_layer = nn.Linear(in_features = action_size + 1, out_features = action_size)
        
        
    def forward(self, x):
        shared_value = torch.relu(self.shared_l1(x))
        value = self.value_layer(shared_value)
        advantage = self.advantage_layer(shared_value)
        #torch.mean(dim, keepdim)->Tensor
        #dim:次元数 keepdim:バッチの次元を維持するかどうか
        q_value = value + advantage - advantage.mean(dim = 1, keepdim = True)
        #torchの型が戻り値
        return q_value
    
    def select_action(self, x):
        x = preprocess(x)
        
        # no_gradの間は計算グラフが構築されない
        with torch.no_grad():
            q = self.forward(x)
        #q.max(index) = torch.max(q, index) -> [[q_value], [index]]
        #バッチに対して考えるのであればtorch.max()[i]は各バッチに対する最大値のリストを返している
        #最大値のインデックスの最大値を取得してnumpyに格納する
        #tensorからscalerをとってきたとしてもtensorの型のままなので変換してやる必要がある
        action = q.max(1)[1][0].numpy()
        return action