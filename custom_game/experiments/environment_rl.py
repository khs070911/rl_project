import pygame
import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
import torch.nn as nn

from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec, CompositeSpec, OneHotDiscreteTensorSpec
from torchrl.data.tensor_specs import TensorSpec
from torchrl.envs import (
    EnvBase, Transform, TransformedEnv, Compose, StepCounter
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp

from environment_naive import CustomEnvironment
from game.code.settings import screen_width, screen_height

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
            low=torch.zeros(4, 140, 140),
            high=torch.ones(4, 140, 140) * 255,
            dtype=torch.float32
        )
        
        self.observation_spec = CompositeSpec(
            obs=self.pixel_spec
        )
        
        self.state_spec = self.observation_spec.clone()
        
        self.action_spec = OneHotDiscreteTensorSpec(n=6)
        
        self.reward = UnboundedContinuousTensorSpec()
        
    def _reset(self, tensordict):
        
        if tensordict is None:
            tensordict = gen_params(batch_size=32)
        
        obs = self.naive_env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        
        state = TensorDict(
            {
                "obs": obs
            }, batch_size=torch.Size([])
        )
        
        return state
    
    def _step(self, act_dict):
        
        act = act_dict["action"].int()
        if act.dim() > 1:
            act = act[0]

        act = act.argmax().item()
        
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
    
class ObsScaling(Transform):
    def _apply_transform(self, obs: torch.Tensor):
        return obs / 255
    
    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict_reset)
    
    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        return BoundedTensorSpec(
            low=0,
            high=1,
            shape=observation_spec.shape,
            dtype=observation_spec.dtype,
            device=observation_spec.device
        )

def make_env(size_rate=0.2, num_stack=4, transform=False, pygame_init=True, show=False):
    if pygame_init:
        pygame.init()
    
    if show:
        screen = pygame.display.set_mode((screen_width, screen_height), flags=pygame.SHOWN)
    else:
        screen = pygame.display.set_mode((screen_width, screen_height), flags=pygame.HIDDEN)
    naive_env = CustomEnvironment(screen, size_rate=size_rate, num_stack=num_stack)
    base_env = EnvironmentForRL(naive_env)
    
    # return base_env
    
    if transform:
        env = TransformedEnv(
            base_env,
            Compose(
                ObsScaling(in_keys=["obs"], out_keys=["obs"]),
                StepCounter(step_count_key="step_count")
            )
        )
    else:
        # env = base_env
        env = TransformedEnv(
            base_env,
            Compose(
                StepCounter(step_count_key="step_count")
            )
        )
    
    return env


def gen_params(g=10.0, batch_size=None) -> TensorDictBase:
    """Returns a ``tensordict`` containing the physical parameters such as gravitational force and torque or speed limits."""
    if batch_size is None:
        batch_size = []
    td = TensorDict(
        {
            "params": TensorDict(
                {
                    "setting": 0.0
                },
                [],
            )
        },
        [],
    )
    if batch_size:
        td = td.expand(batch_size).contiguous()
    return td



# ### for debugging
# pygame.init()
# screen = pygame.display.set_mode((screen_width, screen_height), flags=pygame.HIDDEN) # flags=pygame.HIDDEN pygame.SHOWN
# naive_env = CustomEnvironment(screen, size_rate=0.2, num_stack=4)

# env = EnvironmentForRL(naive_env)
# env = make_env()
# env.transform[0].init_stats(num_iter=2, reduce_dim=0, cat_dim=0)
# print(env.reset())
# check_env_specs(env)
# real_tensordict = env.rollout(3, return_contiguous=True)
# print(real_tensordict[0]["action"])

# pygame.init()
# env = make_env()
# env.transform[0].init_stats(num_iter=10, reduce_dim=0, cat_dim=0)
# print("normalization constant shape:", env.transform[0].loc.shape)

