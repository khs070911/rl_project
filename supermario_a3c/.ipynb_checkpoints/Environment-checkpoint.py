import ast
import gym
from copy import deepcopy
import numpy as np
import pandas as pd

from Reward import reward_func

"""
State : [risk, pp_mtime(0 ~ 60), pp_stime(0 ~ 60), freq(대략 주 발생횟수 혹은 월 발생횟수 같은 통계치) 
"""

class EventEnvironment(gym.Env):
    
    def __init__(self):
        super(EventEnvironment, self).__init__()
        self.episodes = self._initialize()
        
        self.actions_dict = None
        
        self.alpha = 0.5
        self.features, self.scores = self._load_from_file()
        self.current_state = None
        self.best_order = None
        self.is_end = False
        self.action_list = None
        
    def _initialize(self):
        data_path = "data/det_priority_train.csv"
        df = pd.read_csv(data_path)
        episodes = [val['event_id'].tolist() for ids, val in df.groupby('collection_id')]
        
        return episodes
        
        
    def reset(self, seed=None, epi_seed=None):
        
        if seed is not None:
            np.random.seed(seed)
        
        if self.actions_dict is None:
            with open('templates/event_handling_priority/event_state.json', 'r') as f:
                self.actions_dict = ast.literal_eval(f.read())
                
            with open('templates/event_handling_priority/state_event.json', 'r') as f:
                self.reverse_actions_dict = ast.literal_eval(f.read())
                
        if epi_seed is not None:
            random_idx = epi_seed
        else:
            random_idx = np.random.randint(low=0, high=len(self.episodes))
            
        self.action_list = list()
        self.current_state = deepcopy(self.episodes[random_idx])
        self.best_order = self._prioritize(self.current_state)
        init_state = self._preprocess()
        
        return init_state, 0, False
    
    def step(self, action):
        
        action = self.reverse_actions_dict[str(action)]
        done = False
        
        if action in self.current_state:
            
            reward = 0
            self.action_list.append(action)
            
            dup_action_cnt = self.current_state.count(action)
            for _ in range(dup_action_cnt):
                self.current_state.remove(action)
                
            if len(set(self.current_state)) == 1:
                done = True
                self.action_list.append(self.current_state[0])
                ## calculation reward
                reward = self._calculate_reward()
        
        else:
            reward = -10
            done = True
            
        next_state = self._preprocess(done)
            
        return next_state, reward, done
            
    def _preprocess(self, done=False):
        
        row, col = len(self.actions_dict), len(self.features[self.reverse_actions_dict['0']])
        feat_data = np.zeros([row, col])
        
        if done:
            return np.expand_dims(feat_data, axis=0)
        
        for state in self.current_state:
            pos_idx = int(self.actions_dict[state])
            if feat_data[pos_idx, :].sum() > 0:
                continue
            feature = self.features[state]
            feature = list(map(float, feature))
            
            feat_data[pos_idx, :] = feature
            
        return np.expand_dims(feat_data, axis=0)
    
    def _prioritize(self, current_state):
        
        state_score_list = list()
        
        for state in current_state:
            state_score_list.append([state, self.scores[state]])
        
        state_score_list = sorted(state_score_list, key=lambda x: x[1], reverse=True)
        best_order = [state[0] for state in state_score_list]
        
        return best_order
    
    def _calculate_reward(self):
        
        return reward_func(self.best_order, self.action_list)
        
        # reward = 0
        # for evt1, evt2 in zip(self.best_order, self.action_list):
        #     if evt1 == evt2:
        #         reward += 5
        
        # return reward
        
    def _load_from_file(self):
        res_time_dict = open('templates/event_handling_priority/event_response_time.json', 'r').read()
        res_time_dict = ast.literal_eval(res_time_dict)
        
        risk_dict = open('templates/event_handling_priority/event_risk.json', 'r').read()
        risk_dict = ast.literal_eval(risk_dict)
        
        evt_score = dict()
        for key, val in risk_dict.items():
            risk = (val - 5.8) / (7.6 - 5.8)
            res_time = (res_time_dict[key] - 61) / (1497 - 61)
            score = self.alpha * risk + (1 - self.alpha) * (1 - res_time)
            evt_score[key] = score
        
        evt_features = dict()
        for key, val in risk_dict.items():
            risk = val
            mtime = res_time_dict[key] // 60
            stime = res_time_dict[key] % 60
            freq = 0
            
            evt_features[key] = [risk, mtime, stime, freq]
        
        return evt_features, evt_score
    
    def action_space(self):
        if self.actions_dict is None:
            raise Exception("NotResetError: you should run reset method")
        return self.actions_dict
    
    def state_space(self):
        print("This environment has continuous state space!")
        