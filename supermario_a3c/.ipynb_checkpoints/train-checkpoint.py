import os, easydict, random
import numpy as np
from datetime import date
import matplotlib.pyplot as plt

import torch
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

from Agent import SharedAdam, ActorCritic, LearningThread, weights_init
from Reward import ICM

os.environ['OMP_NUM_THREADS'] = '1'
torch.manual_seed(43)
torch.cuda.manual_seed(43)
torch.cuda.manual_seed_all(43)
np.random.seed(43)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(43)

def main(args):
    global_net = ActorCritic(args.kernel_size, args.out_channel, args.hid_size, args.n_actions)
    global_icm = ICM((35, 4), args.n_actions)
    global_net.share_memory()
    global_icm.share_memory()
    optimizer = SharedAdam(list(global_net.parameters()) + list(global_icm.parameters()), lr=2e-6)
    global_episode, global_episode_reward, reward_queue = mp.Value('i', 0), mp.Queue(), mp.Queue()
    
    workers = []
    for i in range(int(mp.cpu_count() * 0.6)):
        w = LearningThread(global_net,
                           global_icm,
                           optimizer,
                           global_episode,
                           global_episode_reward,
                           reward_queue,
                           args,
                           i)
        workers.append(w)
        
    [w.start() for w in workers]
    
    res = []
    while True:
        r = reward_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    
    plt.plot(res)
    plt.ylabel("Moving average episode reward")
    plt.xlabel("Step")
    
    save_date = ''.join(date.today().isoformat().split('-'))[2:]
    plt.savefig('img/reward_curve_{}.png'.format(save_date))
    
    torch.save(global_net.state_dict(), 'model/evt_prior_model_{}.pts'.format(save_date))
    
    [w.join() for w in workers]
    
#     plt.plot(res)
#     plt.ylabel("Moving average episode reward")
#     plt.xlabel("Step")
    
#     save_date = ''.join(date.today().isoformat().split('-'))[2:]
#     plt.savefig('img/reward_curve_{}.png'.format(save_date))
    
    # torch.save(global_net.state_dict(), 'model/evt_prior_model_{}.pts'.format(save_date))
    
if __name__ == "__main__":
    ## set multiprocessing method
    mp.set_start_method('spawn')
    print("MP Start Method : ", mp.get_start_method()) 
    
    ## checking gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ## set argument
    args = easydict.EasyDict(
        {
            ##model hyperparameter
            'kernel_size' : 4,
            'out_channel' : 32,
            'hid_size' : 1024,
            'n_actions' : 35,
            
            ## icm hyperparameter
            'beta' : 0.2,
            'lambda_' : 0.1,
            'eta' : 0.2,
            
            ## model training hyperparameter
            'num_episodes' : 10000,
            'update_gradient' : 10,
            'gamma' : 0.9,
            'epsilon_greedy' : False,
            # 'eps_start' : 0.99,
            # 'eps_end' : 0.05,
            # 'eps_decay' : 5000,
            'device' : device
        }
    )
    
    main(args)