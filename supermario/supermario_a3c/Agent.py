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

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros
from Environment import wrapping_env

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
    
    def __init__(self, in_channel, out_size):
        super(ActorCritic, self).__init__()

        self.mid_network = nn.Sequential(
            nn.Conv2d(in_channel, 32, 3, stride=2), ## floor((84-3)/2 + 1) = 41
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), ## floor((41-3)/2 + 1) = 20
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), ## floor((20-3)/2 + 1) = 9
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), ## floor((9-3)/1 + 1) = 7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128)
        )

        self.pi = nn.Linear(128, out_size)
        self.v = nn.Linear(128, 1)

    def forward(self, x):
        mid = self.mid_network(x)
        # print(mid.shape)

        pi = self.pi(mid)
        v = self.v(mid)

        return pi, v
    
class LearningThread(mp.Process):
    def __init__(self, global_net, optimizer, global_episode_id, global_episode_reward, reward_queue, args, name):
        super(LearningThread, self).__init__()
        self.env = wrapping_env(gym_super_mario_bros.make("SuperMarioBros-1-3-v0"))
        self.name = 'w%02i'%name
        self.global_net = global_net
        self.optimizer = optimizer
        self.local_net = ActorCritic(args.input_dim, args.n_actions)
        
        ## weight initialization
        self.local_net.apply(weights_init)
        
        self.global_episode_id = global_episode_id
        self.global_episode_reward = global_episode_reward
        self.reward_queue = reward_queue
        
        self.num_episodes = args.num_episodes
        self.update_gradient = args.update_gradient
        self.n_actions = args.n_actions
        self.gamma = args.gamma
        self.tau = args.tau
        self.beta = args.beta
        self.device = args.device
        
        self.eps_threshold = 0
        self.step_done = 1
        self.curr_episode = 0
        
    def run(self):

        while self.curr_episode < self.num_episodes:
            obs = self.env.reset()
            buffer_pi, buffer_v, buffer_r = [], [], []
            epi_reward = 0
            obs = torch.tensor(obs.__array__())
            obs = obs.unsqueeze(0).float()
            while True:

                ## select action over state
                pi, v = self.local_net(obs)
                prob = F.softmax(pi, dim=1)
                m = Categorical(prob)
                action = m.sample()

                ## apply action selected by model
                next_obs, reward, done, info = self.env.step(action.item()) ## 애초에 reward가 이동거리인 듯
                epi_reward += reward
                
                next_obs = torch.tensor(next_obs.__array__()).unsqueeze(0).float()
                
                buffer_pi.append(m.log_prob(action))
                buffer_v.append(v)
                buffer_r.append(reward)
                
                if self.step_done % self.update_gradient == 0 or done: 
                    self.update_global_gradient_and_pull_parameter(next_obs, done, buffer_pi, buffer_v, buffer_r)
                    buffer_pi, buffer_v, buffer_r = [], [], []
                
                if done:
                    break
                
                obs = next_obs
                self.step_done += 1
            self.curr_episode += 1
            
            record(self.global_episode_id, self.global_episode_reward, epi_reward, self.name, self.eps_threshold)

        self.reward_queue.put(None)
        
    def update_global_gradient_and_pull_parameter(self, next_obs, done, buffer_pi, buffer_v, buffer_r):

        ## atcor & critic loss calculation
        if done:
            next_value = 0
        else:
            next_value = self.local_net(next_obs)[-1].item()
        
        ### version 1 : GAE ####
        gae = 0
        R = next_value
        buffer_gae = []
        buffer_v_target = []
        for r, v in list(zip(buffer_r, buffer_v))[::-1]:
            gae = gae * self.tau * self.gamma
            gae = gae + r + self.gamma * R - v
            R = v
            buffer_gae.append(gae)

            next_value = r + self.gamma * next_value
            buffer_v_target.append(next_value)

        # #### version 2 : Basic ####
        # buffer_v_target = []
        # for r in buffer_r[::-1]:
        #     next_value = r + self.gamma * next_value
        #     buffer_v_target.append(next_value)
                
        buffer_v_target.reverse()
        buffer_gae.reverse()
        
        buffer_pi = torch.cat(buffer_pi).view(-1)
        buffer_v = torch.cat(buffer_v).view(-1)
        buffer_v_target = torch.tensor(buffer_v_target)
        buffer_gae = torch.tensor(buffer_gae)
        
        td = buffer_v_target - buffer_v
        c_loss = td.pow(2)
        
        exp_v = buffer_pi * buffer_gae.detach() ## detach의 중요성을 잘 알자 -> loss 구할 시 관련없는 텐서는 detach로 새로 구해야 함
        a_loss = -exp_v
        
        total_loss =  (c_loss + a_loss).sum() - self.beta * ((torch.exp(buffer_pi) * buffer_pi).sum())
        
        self.optimizer.zero_grad()
        total_loss.backward()
        for lp, gp in zip(self.local_net.parameters(), self.global_net.parameters()):
            gp._grad = lp._grad
        self.optimizer.step()
        
        self.local_net.load_state_dict(self.global_net.state_dict())

def record(global_ep, global_ep_r, ep_r, res_queue, name, eps_threshold):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
        "| Eps: %.2f" % eps_threshold
    )
    
# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal_(m.weight)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight)