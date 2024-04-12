import pygame, time
import torch
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, OneHotCategorical

from environment_rl import make_env

def test():
    
    env = make_env(transform=True, show=True)
    
    actor_net = torch.load("./models/actor.pt")
    policy_module = TensorDictModule(
            actor_net,
            in_keys=["obs"],
            out_keys=["logits"]
    )
    
    actor = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["logits"],
        distribution_class=OneHotCategorical,
        return_log_prob=True
    )
    
    done = False
    obs = env.reset()
    
    while not done:
        with torch.no_grad():
            action_dict = actor(obs)
        next_dict = env.step(action_dict)
        
        print(next_dict["next", "reward"])
        done = next_dict["next", "done"].item()
        
    
    # print()
        
        

if __name__ == "__main__":
    test()