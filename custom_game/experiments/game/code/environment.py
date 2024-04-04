import gym
import cv2
import numpy as np
import pygame, sys
import pygame, sys, random
from settings import screen_height, screen_width
from level import Level


class CustomEnvironment:
    
    def __init__(self):
        
        # game attributes
        self.max_level = 0
        self.max_health = 100
        self.cur_health = 100
        self.coins = 0
        
        