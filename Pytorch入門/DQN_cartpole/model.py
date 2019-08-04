import torch
import torch.nn as nn

def preprocess(x):
    x = torch.tensor(x, dtype = torch.float).unsqueeze(0)
    return x

class QFunc(nn.Module):
    def __init__(self, obs_size = 4, action_size = 2):
        super(QFunc, self).__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        
        self.l1 = nn.Linear(in_features = obs_size, out_features = 50)
        self.l2 = nn.Linear(in_features = 50, out_features = action_size)
        
    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = self.l2(x)
        #torchが戻り値
        return x
    
    def select_action(self, x):
        x = preprocess(x)
        
        # no_gradの間は計算グラフが構築されない
        with torch.no_grad():
            q = self.forward(x)
        #q.max(1) = torch.max(q, 1) -> [q_value, index]
        #最大値のインデックスの最大値を取得してnumpyに格納する
        action = q.max(1)[1][0].numpy()
        return action