import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

import numpy as np
import torch
from torchvision import transforms as T
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY


class SkipFrame(gym.Wrapper):
    ## 연속된 프레임 간에는 정보 변화(?)가 거의 없기 때문에 두 데이터가 거의 비슷하다. 따라서, 중간 프레임을 생략해도 정보 손실이 없을 것이다. 그렇게 해서 얻은 n번째 프레임은 생략 프레임에서 얻은 보상까지 포함한다.
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)] ## 255로 나눠서 normalization
        )
        observation = transforms(observation).squeeze(0)
        return observation


def wrapping_env(env):
    
    env = JoypadSpace(env, RIGHT_ONLY) #SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
    
    # Apply Wrappers to environment -> 필요한 전처리 기능을 환경에 넣어 감싸준 것(?)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    
    return env