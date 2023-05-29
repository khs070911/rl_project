
import os
import time
from pathlib import Path
import datetime
import numpy as np
import argparse
import gym
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

from memory import Memory
from model import DuelDQN
from ddql import DoubleQL
from utils import get_logger
from normalized_env import wrapping_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.batch_size = 128
    args.gamma = 0.999
    args.eps_start = 1.0
    args.eps_end = 0.05
    args.eps_decay = 20000
    args.num_episodes = 3000
    args.target_update = 20
    args.restore = False #True
    
    logger = get_logger("logs/training.log")
    
    env = wrapping_env(gym_super_mario_bros.make("SuperMarioBros-1-1-v0"))
    args.n_actions = env.action_space.n
    
    init_obs = env.reset()
    if not args.restore:
        policy_net = DuelDQN((4, 84, 84), args.n_actions)
    else:
        restore_path = os.path.join('./models', 'old2', 'model_mario.pt')
        policy_net = torch.load(restore_path)
        logger.info("[Model Restore]")
    target_net = DuelDQN((4, 84, 84), args.n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0008)
    memory = Memory(100000)
    
    ddql = DoubleQL(env=env,
                memory=memory,
                target_network=target_net,
                q_network=policy_net,
                optimizer=optimizer,
                criterion="huber",
                args=args,
                device="cpu")
    
    loss_list, score_list = ddql.train(logger)
    
    logger.info("[Last Model Save]")
    ddql.last_save()
    
    logger.info("[Loss, Score Graph Save]")
    # plt.subplot(2, 1, 1)
    plt.plot([idx for idx in range(len(loss_list))], loss_list)
    plt.title('Loss Graph')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.savefig('./plot/loss_graph.png')
    
    plt.cla()
    plt.plot([idx for idx in range(len(score_list))], score_list)
    plt.title('Score Graph')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig('./plot/score_graph.png')
    
    