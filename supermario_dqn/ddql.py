import os
import random
import math
from itertools import count
from collections import namedtuple
import numpy as np
import torch
import torch.nn.functional as F

class DoubleQL(object):
    def __init__(self, env, memory, target_network, q_network, optimizer, criterion, args, device):
        self.env = env
        self.memory = memory
        self.target_network = target_network
        self.q_network = q_network
        self.optimizer = optimizer
        self.criterion = criterion
        
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay = args.eps_decay
        self.n_actions = args.n_actions
        self.num_episodes = args.num_episodes
        self.target_update = args.target_update
        self.device = device
        
        self.step_done = 0
        self.model_dir = './models'
    
    def select_action(self, x):

        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1 * self.step_done / self.eps_end)
            
        self.step_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                self.q_network.eval()
                return self.q_network(x).max(1)[1]
            
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device)
    
    def update_network(self):
        
        if self.memory.tree.n_entries < self.batch_size * 2:
            return
        
        transition, idice, is_weights = self.memory.sample(self.batch_size)
        
        Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
        
        try:
            batch = Transition(*zip(*transition))
        except:
            transition, idice, is_weights = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transition))
            print("Exception occur")
        
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action).view(-1, 1)
        reward_batch = torch.cat(batch.reward).view(-1, 1)
        done_batch = torch.cat(batch.done).view(-1, 1)
        
        self.q_network.train()
        self.target_network.train()
        
        next_q_value = self.q_network(next_state_batch)
        best_action = torch.argmax(next_q_value, axis = 1)
        target_q_value = self.target_network(next_state_batch)[np.arange(0, self.batch_size), best_action]
        target_q_value = target_q_value.view(-1, 1)
        target_value = reward_batch + (target_q_value * self.gamma) * (1 - done_batch.int())
        
        state_action_values = self.q_network(state_batch).gather(1, action_batch)
        
        if self.criterion == "mse":
            loss = F.mse_loss(state_action_values, target_value) * torch.tensor(is_weights)
        else:
            loss = F.smooth_l1_loss(state_action_values, target_value) * torch.tensor(is_weights)
        
        loss = loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        errors = torch.abs(state_action_values - target_value).data.numpy()
        
        for i in range(self.batch_size):
            idx = idice[i]
            self.memory.update(idx, errors[i])
            
        return loss.item()
    
    def compute_error(self, obs, action, next_obs, reward, done):
        
        with torch.no_grad():
            self.q_network.eval()
            self.target_network.eval()
            state_values = self.q_network(obs)[0][action.item()]
            
            best_action = self.q_network(next_obs).argmax()
            target_value = self.target_network(next_obs)[0][best_action.item()]
            target_value = reward + self.gamma * target_value * (1 - done.int())
            
        error = torch.abs(target_value - state_values)
        return error.item()
    
    def train(self, logger):
        
        logger.info("<<<< Train Start >>>>")
        
        mean_score_criterion = 1600
        score_up_size = 20

        episode_loss_list = []
        episode_reward_list = []
        # Initialize the environment and state
        for episode in range(self.num_episodes):
            obs = self.env.reset()
            obs = torch.tensor(obs.__array__(), device=self.device).unsqueeze(0)
            
            episode_reward = 0
            episode_loss = 0
            for t in count():
                action = self.select_action(obs)
                next_obs, reward, done, _ = self.env.step(action.item())
                
                next_obs = torch.tensor(next_obs.__array__(), device=self.device).unsqueeze(0)
                reward = torch.tensor([reward], device=self.device)
                done = torch.tensor([done], device=self.device)
                action = torch.tensor([action.item()], device=self.device)
                
                error = self.compute_error(obs, action, next_obs, reward, done)
                self.memory.push(error, [obs, action, next_obs, reward, done])
                
                obs = next_obs
                episode_reward += reward.item()
                
                q_loss = self.update_network()
                
                if q_loss is not None:
                    episode_loss += q_loss
                
                if done:
                    break
                
                # self.env.render()
            
            episode_loss_mean = episode_loss / t
            ## logging
            logger.info("Episode : %d, reward : %d, loss : %f"%(episode, episode_reward, episode_loss_mean))
            episode_loss_list.append(episode_loss_mean)
            episode_reward_list.append(episode_reward)
            
            if episode % self.target_update == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                
            if np.mean(episode_reward_list[-100:]) >= mean_score_criterion and np.mean(episode_reward_list[-10:]) >= mean_score_criterion:
                mean_score_criterion += score_up_size
                self.save()
                logger.info("[Model Save and save threshold : %d"%(mean_score_criterion))
                
        return episode_loss_list, episode_reward_list
    
    def save(self):
        model_path = os.path.join(self.model_dir, "model_mario.pt")
        torch.save(self.q_network, model_path)
        
    def last_save(self):
        model_path = os.path.join(self.model_dir, "last_model_mario.pt")
        torch.save(self.q_network, model_path)