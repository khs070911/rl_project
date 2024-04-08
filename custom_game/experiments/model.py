import pygame
from environment_rl import make_env

import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor

actor_net = nn.Sequential(
    nn.LazyConv2d(32, kernel_size=3, stride=3),
    nn.ELU(),
    nn.LazyConv2d(32, kernel_size=3, stride=3),
    nn.ELU(),
    nn.LazyConv2d(32, kernel_size=3, stride=3),
    nn.Flatten(),
    nn.LazyLinear(64),
    nn.ELU(),
    nn.LazyLinear(6),
    nn.Softmax()
)

policy_module = TensorDictModule(
    actor_net,
    in_keys=["obs"]
)

pygame.init()
env = make_env()
env.transform[0].init_stats(num_iter=10, reduce_dim=0, cat_dim=0)

actor = ProbabilisticActor(
    policy_module,
    spec=env.action_spec,
    in_keys=["action"],
)