### Reward Function : 모델이 예측한 우선순위와 정답과 비교하여 보상 부여 
### Reward Func. = DCG - Median Index DCG

### intrinsic reward by ICM(Intrinsic Curiosity Module)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def reward_func(target, prediction):
    
    mdcg = cal_dcg(target, target, median=True)
    dcg = cal_dcg(target, prediction)
    idcg = cal_dcg(target, target)
    
    return (dcg - mdcg)/(idcg - mdcg)

def cal_dcg(target, prediction, median=False):
    
    dcg = 0
    for i, item in enumerate(prediction):
        if median:
            relevance_score = (len(target) + 1)/2
        else:
            relevance_score = (len(target) - target.index(item))
        dcg += relevance_score/(math.log10(i + 2))
        
    return dcg


class ICM(nn.Module):
    def __init__(self, input_size, output_size):
        super(ICM, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        out_channels = 32
        feature_mid = 32 * out_channels
        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=4
            ),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(feature_mid, 256),
        )
        
        self.inverse_net = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_size)
        )
        
        self.forward_net = nn.Sequential(
            nn.Linear(output_size + 256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256)
        )
        
    def forward(self, inputs):
        state, next_state, action = inputs # action is one-hot vector
        
        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
        
        # predict action
        pred_action = torch.cat([encode_state, encode_next_state], 1)
        pred_action = self.inverse_net(pred_action)
        
        # predict next state
        pre_next_state_feature = torch.cat([encode_state, action], 1)
        pre_next_state_feature = self.forward_net(pre_next_state_feature)
        
        return encode_next_state, pre_next_state_feature, pred_action
        