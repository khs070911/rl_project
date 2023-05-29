import math
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical
import torch.backends.cudnn as cudnn

from Environment import EventEnvironment
from Reward import ICM

torch.manual_seed(43)
torch.cuda.manual_seed(43)
torch.cuda.manual_seed_all(43)
np.random.seed(43)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(43)

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                
class ActorCritic(nn.Module):
    def __init__(self, kernel_size, out_channel, hid_size, out_size):
        super(ActorCritic, self).__init__()
        
        self.seq = nn.Sequential(
                                nn.Conv2d(in_channels=1, out_channels=out_channel, kernel_size=kernel_size),
                                nn.ELU(),
                                nn.Flatten(),
                                nn.Linear(out_channel * 32, hid_size),
                                nn.ELU()
                                )
        
        self.pi = nn.Linear(hid_size, out_size)
        self.v = nn.Linear(hid_size, 1)
        
    def forward(self, x):
        mid = self.seq(x)
        
        pi = self.pi(mid)
        v = self.v(mid)
        
        return pi, v
    
class LearningThread(mp.Process):
    def __init__(self, global_net, global_icm, optimizer, global_episode_id, global_episode_reward, reward_queue, args, name):
        super(LearningThread, self).__init__()
        self.env = EventEnvironment()
        self.name = 'w%02i'%name
        self.global_net = global_net
        self.global_icm = global_icm
        self.optimizer = optimizer
        self.local_net = ActorCritic(args.kernel_size, args.out_channel, args.hid_size, args.n_actions)
        self.local_icm = ICM((35, 4), args.n_actions)
        
        ## weight initialization
        self.local_net.apply(weights_init)
        self.local_icm.apply(weights_init)
        
        self.global_episode_id = global_episode_id
        self.global_episode_reward = global_episode_reward
        self.reward_queue = reward_queue
        
        self.num_episodes = args.num_episodes
        self.update_gradient = args.update_gradient
        self.gamma = args.gamma
        self.n_actions = args.n_actions
        self.epsilon_greedy = args.epsilon_greedy
        # self.eps_start = args.eps_start
        # self.eps_end = args.eps_end
        # self.eps_decay = args.eps_decay
        self.beta = args.beta
        self.lambda_ = args.lambda_
        self.eta = args.eta
        self.device = args.device
        
        self.eps_threshold = 0
        self.step_done = 1
        self.curr_episode = 0
        
    def run(self):
        
        while self.curr_episode < self.num_episodes:
            obs,_,_ = self.env.reset(epi_seed = self.curr_episode % len(self.env.episodes))
            buffer_s, buffer_a, buffer_r, buffer_s_, buffer_a_onehot = [], [], [], [], []
            epi_reward = 0
            obs = torch.tensor(obs)
            obs = obs.unsqueeze(0).float()
            while True:
                if self.epsilon_greedy:
                    action = self.select_action_with_epsilon(obs)
                else:
                    action = self.select_action(obs)
                next_obs, reward, done = self.env.step(action)
                epi_reward += reward
                
                ## add intrinsic reweard
                next_obs = torch.tensor(next_obs).unsqueeze(0).float()
                action_onehot = F.one_hot(torch.tensor([action]), num_classes=self.n_actions).float()
                intrinsic_reward = self.compute_intrinsic_reward(obs, next_obs, action_onehot)
                reward += intrinsic_reward
                
                buffer_s.append(obs)
                buffer_a.append(action)
                buffer_r.append(reward)
                buffer_s_.append(next_obs)
                buffer_a_onehot.append(action_onehot)
                
                if done: #self.step_done % self.update_gradient == 0 or 
                    self.update_global_gradient_and_pull_parameter(next_obs, done, buffer_s, buffer_a, buffer_r, buffer_s_, buffer_a_onehot)
                    buffer_s, buffer_a, buffer_r, buffer_s_, buffer_a_onehot = [], [], [], [], []
                    
                    self.global_episode_reward.put(epi_reward)
                    break
                
                obs = next_obs
                self.step_done += 1
            self.curr_episode += 1
            
            save_and_record(self.global_episode_id, self.global_episode_reward, self.reward_queue, self.eps_threshold, self.name)

        self.reward_queue.put(None)
                
    
    def select_action(self, obs):
        
        with torch.no_grad():
            pi, _ = self.local_net(obs)
            mask = (obs.sum(-1).squeeze() == 0)
            prob = F.softmax(pi + mask * (-1e9), dim=1)
            m = Categorical(prob)
            
        return m.sample().numpy()[0]
    
    def select_action_with_epsilon(self, obs):
        sample = random.random()
        self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1 * self.step_done/self.eps_decay)
        
        if sample > self.eps_threshold:
            return self.select_action(obs)
        else:
            weights = (obs.sum(-1).squeeze() > 0).int().numpy()
            random_action = random.choices(range(self.n_actions), weights=weights)
            return random_action[0]
        
    def compute_intrinsic_reward(self, obs, next_obs, action_onehot):
        # obs = torch.tensor(obs).unsqueeze(0)
        
        real_next_state_feature, pre_next_state_feature, _ = self.local_icm([obs, next_obs, action_onehot])
        intrinsic_reward = self.eta * F.mse_loss(real_next_state_feature, pre_next_state_feature).mean()
        return intrinsic_reward.item()
        
    def update_global_gradient_and_pull_parameter(self, next_obs, done, buffer_s, buffer_a, buffer_r, buffer_s_, buffer_a_onehot):
        
        ## atcor & critic loss calculation
        if done:
            next_value = 0
        else:
            next_value = self.local_net(next_obs)[-1]
            
        buffer_v_target = []
        for r in buffer_r[::-1]:
            next_value = r + self.gamma * next_value
            try:
                buffer_v_target.append(next_value.item())
            except:
                buffer_v_target.append(next_value)
                
        buffer_v_target.reverse()
        
        buffer_s = torch.tensor(np.vstack(buffer_s))
        buffer_a = torch.tensor(np.array(buffer_a)).view(-1, 1)
        buffer_r = torch.tensor(np.array(buffer_r)).view(-1, 1)

        buffer_v_target = torch.tensor(np.array(buffer_v_target)).view(-1, 1)
        
        self.local_net.train()
        pi, values = self.local_net(buffer_s)
        adv = buffer_v_target - values
        c_loss = adv.pow(2)
        
        prob = F.softmax(pi, dim=1)
        m = Categorical(prob)
        exp_v = m.log_prob(buffer_a) * adv.detach().squeeze()
        a_loss = -exp_v
        
        ## icm model loss calculation
        buffer_s_ = torch.tensor(np.vstack(buffer_s_))
        buffer_a_onehot = torch.tensor(np.vstack(buffer_a_onehot))
        
        enc_next, enc_state, pred_action = self.local_icm([buffer_s, buffer_s_, buffer_a_onehot])
        fwd_loss = F.mse_loss(enc_next, enc_state)
        inv_loss = F.cross_entropy(pred_action, buffer_a.view(-1))
        
        total_loss = (self.lambda_ * (c_loss + a_loss) + self.beta * fwd_loss + (1 - self.beta) * inv_loss).mean()
        
        self.optimizer.zero_grad()
        total_loss.backward()
        for lp, gp in zip(self.local_net.parameters(), self.global_net.parameters()):
            gp._grad = lp._grad
        for lp, gp in zip(self.local_icm.parameters(), self.global_icm.parameters()):
            gp._grad = lp._grad
        self.optimizer.step()
        
        self.local_net.load_state_dict(self.global_net.state_dict())
        self.local_icm.load_state_dict(self.global_icm.state_dict())
        
def save_and_record(global_ep, ep_r, res_queue, eps, name):
    with global_ep.get_lock():
        global_ep.value += 1
    
    epi_idx = global_ep.value
    if epi_idx % 100 == 0:
        moving_reward = 0
        for i in range(1, 101):
            try:
                r = ep_r.get_nowait()
                moving_reward += r
            except:
                break
        
        moving_reward = moving_reward/i
        if moving_reward > 0:
            res_queue.put(moving_reward)
        
        print(
            name,
            "Ep:", epi_idx,
            "| Ep_r_moving : %.4f" % moving_reward,
            "| Epsilon : %.4f" % eps
        )
    
# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.1)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.1)