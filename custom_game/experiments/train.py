import pygame
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

import torch
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules.tensordict_module import EGreedyModule
from tensordict.nn import TensorDictSequential

from environment_rl import make_env
from models import make_actor_critic

def train(args):
    
    # arguments
    num_workers = args["num_workers"]
    total_frames = args["total_frames"]
    frames_per_batch = args["frames_per_batch"]
    sub_batch_size = args["sub_batch_size"]
    
    gamma = args["gamma"]
    lmbda = args["lmbda"]
    clip_epsilon = args["clip_epsilon"]
    entropy_eps = args["entropy_eps"]
    lr = args["lr"]
    num_epochs = args["num_epochs"]
    grad_clip = args["grad_clip"]
    
    # actor-critic
    pygame.init()
    env = make_env(pygame_init=False)
    actor, critic = make_actor_critic(env.action_spec)
    
    # collector
    collector = MultiSyncDataCollector(
        create_env_fn=[make_env]*num_workers,
        # env,
        policy=actor,
        total_frames=total_frames,
        frames_per_batch=frames_per_batch,
        reset_at_each_iter=True,
        split_trajs=False
    )
    
    # replay buffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )
    
    # setting for training
    adv_module = GAE(gamma=gamma, lmbda=lmbda, value_network=critic, average_gae=True)
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        critic_coef=1.0,
        loss_critic_type="smooth_l1"
    )
    
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=1e-6, step_size_up=5, max_lr=1e-4, gamma=0.9, mode="exp_range")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95**epoch)
    
    # training
    logs = defaultdict(list)
    optimizer.zero_grad()
    # pbar = tqdm(total=total_frames)
    curr_step = 0
    
    for i, tensordict_data in enumerate(collector):
        
        for _ in range(num_epochs):
            
            adv_module(tensordict_data)
            replay_buffer.extend(tensordict_data)
            
            for _ in tqdm(range(frames_per_batch // sub_batch_size)):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_dict = loss_module(subdata)
                loss_value = (
                    loss_dict["loss_objective"] 
                    + loss_dict["loss_critic"]
                    + loss_dict["loss_entropy"]
                )
                
                loss_value.backward()
                
                # gradient clipping
                if bool(grad_clip):
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), grad_clip)

                optimizer.step()
                optimizer.zero_grad()
                
        # pbar.update(tensordict_data.numel())
        
        if i % 1 == 0:
            with torch.no_grad():
                eval_rollout = env.rollout(300, actor)
                
                avg_reward = eval_rollout["next", "reward"].sum().item()
                step_count = eval_rollout["step_count"].max().item()
                
                logs["reward"].append(avg_reward)
                logs["max_step"].append(step_count)
                
                curr_step += tensordict_data.numel()
                
                desc = "Current Step: {} , Reward(sum): {}, Max Step: {}".format(curr_step, avg_reward, step_count)
                print(desc)
                # pbar.set_description(desc=desc)
        
        # avg_reward = tensordict_data["next", "reward"].mean().item()
        # step_count = tensordict_data["step_count"].max().item()
        
        # logs["reward"].append(avg_reward)
        # logs["max_step"].append(step_count)
        # desc = "Reward(avg) : {}, Max Step : {}".format(avg_reward, step_count)
        # pbar.set_description(desc=desc)
        
            scheduler.step()
        
    collector.shutdown()
    
    # model saving
    torch.save(actor, "./models/actor.pt")
    torch.save(critic, "./models/critic.pt")
    
    return logs
    

if __name__ == "__main__":
    args = {
        "num_workers" : 4,
        "total_frames" : 20000,
        "frames_per_batch" : 1000,
        "sub_batch_size" : 10,
        
        "gamma" : 0.99,
        "lmbda" : 0.95,
        "entropy_eps" : 1e-2,
        "clip_epsilon" : 0.9,
        "lr" : 1.41e-5,
        "num_epochs" : 2,
        "grad_clip" : 0
    }
    
    logs = train(args)
    
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(logs["reward"])
    plt.title("Reward(Avg)")
    
    plt.subplot(1, 2, 2)
    plt.plot(logs["max_step"])
    plt.title("Max Step Count")
    
    plt.savefig("result.png")
    