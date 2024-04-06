import pygame
import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
import torch.nn as nn

from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec, CompositeSpec
from torchrl.envs import (
    EnvBase, Transform, TransformedEnv
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp

from environment_naive import CustomEnvironment

class EnvironmentForRL(EnvBase):
    
    def __init__(self, naive_env, seed=None, device="cpu"):
        
        self.naive_env = naive_env
        
        super().__init__(device=device, batch_size=[])
        
        self._make_spec()
        self._set_seed(seed=seed)
        
    def _set_seed(self, seed):
        seed = torch.randint(0, 255, size=(1,)).item()
        self.rng = torch.manual_seed(seed)
    
    def _make_spec(self):
        
        self.pixel_spec = BoundedTensorSpec(
            low=torch.zeros(4, 140, 240),
            high=torch.ones(4, 140, 240) * 255,
            dtype=torch.float32
        )
        
        self.observation_spec = CompositeSpec(
            obs=self.pixel_spec
        )
        
        self.state_spec = self.observation_spec.clone()
        
        self.action_spec = DiscreteTensorSpec(n=6)
        
        self.reward = UnboundedContinuousTensorSpec(dtype=torch.int32)
        
    def _reset(self, tensordict):
        
        obs = self.naive_env.reset()
        obs = torch.tensor(obs)
        
        state = TensorDict(
            {
                "obs": obs
            }, batch_size=torch.Size([])
        )
        
        return state
    
    def _step(self, act_dict):
        
        act = act_dict["action"]
        next_obs, reward, done = self.naive_env.step(act)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done)
        
        out = TensorDict(
            {
                "obs": next_obs,
                "reward": reward,
                "done" : done
            }, batch_size=torch.Size([])
        )
        
        return out
    
# ### for debugging
# pygame.init()
# screen = pygame.display.set_mode((1200, 700), flags=pygame.HIDDEN) # flags=pygame.HIDDEN pygame.SHOWN
# naive_env = CustomEnvironment(screen, size_rate=0.2, num_stack=4)

# env = EnvironmentForRL(naive_env)
# # check_env_specs(env)
# real_tensordict = env.rollout(3, return_contiguous=True)
# print(real_tensordict[0]["action"])