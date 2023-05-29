import os, easydict, random
import numpy as np
from datetime import date
import matplotlib.pyplot as plt

import torch
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

from Agent import SharedAdam, ActorCritic, LearningThread
import gym_super_mario_bros
from Environment import wrapping_env

os.environ['OMP_NUM_THREADS'] = '1'

def main(args):
    global_net = ActorCritic(args.input_dim, args.n_actions)
    global_net.share_memory()
    optimizer = SharedAdam(global_net.parameters(), lr=1e-4)
    global_episode, global_episode_reward, reward_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    
    workers = []
    for i in range(int(mp.cpu_count() * 1)):
        w = LearningThread(global_net,
                           optimizer,
                           global_episode,
                           global_episode_reward,
                           reward_queue,
                           args,
                           i)
        workers.append(w)
        
    [w.start() for w in workers]
    
    # res = []
    # while True:
    #     r = reward_queue.get()
    #     if r is not None:
    #         res.append(r)
    #     else:
    #         break
    
    [w.join() for w in workers]
    
    # plt.plot(res)
    # plt.ylabel("Moving average episode reward")
    # plt.xlabel("Step")
    
    save_date = ''.join(date.today().isoformat().split('-'))[2:]
    # plt.savefig('img/reward_curve_{}.png'.format(save_date))
    
    torch.save(global_net.state_dict(), 'model/supermario_model_{}.pts'.format(save_date))
    
if __name__ == "__main__":
    ## environment
    env = wrapping_env(gym_super_mario_bros.make("SuperMarioBros-1-3-v0"))

    ## set multiprocessing method
    mp.set_start_method('fork')
    print("MP Start Method : ", mp.get_start_method()) 
    
    ## checking gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ## set argument
    args = easydict.EasyDict(
        {
            ##model hyperparameter
            'input_dim' : env.reset().__array__().shape[0],
            'n_actions' : env.action_space.n,
            
            ## model training hyperparameter
            'num_episodes' : 10000,
            'update_gradient' : 80,
            'gamma' : 0.9,
            'tau' : 0.99,
            'beta' : 0.01,
            'device' : device
        }
    )
    
    main(args)