# import pygame
# from environment_rl import make_env

import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, OneHotCategorical, ValueOperator
# from torchrl.collectors import aSyncDataCollector, SyncDataCollector,  MultiSyncDataCollector

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

actor_net = nn.Sequential(
    nn.Conv2d(4, 32, kernel_size=3, stride=3),
    nn.ELU(),
    nn.Conv2d(32, 32, kernel_size=3, stride=3),
    nn.ELU(),
    nn.Conv2d(32, 32, kernel_size=3, stride=3),
    nn.Flatten(),
    Reshape(shape=(-1, 1280)),
    nn.Linear(1280, 64),
    nn.ELU(),
    nn.Linear(64, 6),
)

value_net = nn.Sequential(
    nn.Conv2d(4, 32, kernel_size=3, stride=3),
    nn.ELU(),
    nn.Conv2d(32, 32, kernel_size=3, stride=3),
    nn.ELU(),
    nn.Conv2d(32, 32, kernel_size=3, stride=3),
    nn.Flatten(),
    Reshape(shape=(-1, 1280)),
    nn.Linear(1280, 64),
    nn.ELU(),
    nn.Linear(64, 1),
)

def make_actor_critic(action_spec):
    

    policy_module = TensorDictModule(
            actor_net,
            in_keys=["obs"],
            out_keys=["logits"]
        )
    
    actor = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=["logits"],
        distribution_class=OneHotCategorical,
        return_log_prob=True
    )

    critic = ValueOperator(
        module=value_net,
        in_keys=["obs"]
    )
    
    return actor, critic


# if __name__ == "__main__":
    
#     pygame.init()

#     policy_module = TensorDictModule(
#         actor_net,
#         in_keys=["obs"],
#         out_keys=["logits"]
#     )

#     env = make_env(transform=True)

#     actor = ProbabilisticActor(
#         module=policy_module,
#         spec=env.action_spec,
#         in_keys=["logits"],
#         distribution_class=OneHotCategorical
#     )
#     obs = env.reset()
#     actor(obs)

#     collector =  MultiSyncDataCollector(
#         create_env_fn=[make_env, make_env],
#         policy=actor,
#         total_frames=200,
#         frames_per_batch=100,
#         reset_at_each_iter=False
#     )
    
#     print(collector.num_threads)
    
#     for i, data in enumerate(collector):
#         pass
    
#     print(data)

#     collector.shutdown()