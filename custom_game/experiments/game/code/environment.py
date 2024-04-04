import gym
import cv2
import numpy as np
import pygame, sys
import pygame, sys, random
from settings import screen_height, screen_width
from level import Level
from ui import UI


class CustomEnvironment:
    
    def __init__(self, screen):
        
        self.screen = screen
        
        # game attributes
        self.max_level = 0
        self.max_health = 100
        self.cur_health = 100
        self.coins = 0
        
        # user interface
        self.ui = UI(screen)
        
        # level setting
        self.current_level = 0
        
    def create_level(self, level):
        self.level = Level(level, self.screen, self.create_overworld,self.change_coins,self.change_health)
        
    def change_coins(self, amount):
        self.coins += amount
        
    def change_health(self, amount):
        self.cur_health += amount
        
    def check_game_over(self):
        if self.cur_health <= 0:
            self.cur_health = 100
            self.coins = 0
            self.level = Level(self.current_level, self.screen, self.create_overworld,self.change_coins,self.change_health)
