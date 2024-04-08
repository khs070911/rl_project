import sys

sys.path.append("game/code") ## Mac
sys.path.append("game\code") ## Window

import cv2
import numpy as np
import pygame, sys, random
from game.code.level import Level
from game.code.ui import UI


class CustomEnvironment:
    
    def __init__(self, screen, num_stack=4, size_rate=0.2):
        
        # Pygame setup
        self.screen = screen
        self.clock = pygame.time.Clock()
        
        # game attributes
        self.max_level = 0
        self.max_health = 100
        self.cur_health = 100
        self.coins = 0
        
        # user interface
        self.ui = UI(screen)
        
        # level setting
        self.current_level = 0
        
        # for observation setting
        self.num_stack = num_stack
        self.size_rate = size_rate
        
    def create_level(self, level):
        self.level = Level(level, self.screen, self.change_coins,self.change_health)
        
    def change_coins(self, amount):
        self.coins += amount
        
    def change_health(self, amount):
        self.cur_health += amount
        
    def check_game_over(self):
        if self.cur_health <= 0:
            self.cur_health = 100
            self.coins = 0
            self.level = Level(self.current_level, self.screen, self.create_overworld,self.change_coins,self.change_health)
            
    def run(self):
        check_death, check_win, check_coin, check_enemy, kill_enemy = self.level.run()
        
        self.ui.show_health(self.cur_health, self.max_health)
        self.ui.show_coins(self.coins)
        
        self.pygame_update()
        
        return check_death, check_win, check_coin, check_enemy, kill_enemy
    
    def reset(self):
        
        self.cur_health = 100
        self.coins = 0
        
        self.create_level(self.current_level)
        
        while not self.level.player.sprite.on_ground:
            self.run()
        
        # set obs
        img = self.get_display_img()
        
        # set distance
        self.goal_dist = self.level.get_player_from_goal()
        
        # # pygame update
        # self.pygame_update()
        
        return img
    
    def get_display_img(self):
        
        stack_img = []
        # self.level.player.sprite.get_input("stop", False)
        
        for i in range(self.num_stack):
            self.run()
            
            # state
            screen = pygame.display.get_surface()
            img = pygame.surfarray.array3d(screen)
            img = img.transpose([1, 0, 2])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, dsize=(0,0), fx=self.size_rate, fy=self.size_rate, interpolation=cv2.INTER_LINEAR)
            
            stack_img.append([img])
            
        img = np.concatenate(stack_img)
        # img = img.transpose([1, 2, 0])
        
        return img
    
    def step(self, act):
        """
            act: 
                no jump : 0 = stop, 1 = right, 2 = left
                jump : 3 = stop, 4 = right, 5 = left
        """
        
        space = act > 2
        if space:
            act -= 3
        
        if act == 1:
            act = "right"
        elif act == 2:
            act == "left"
        else:
            act = "stop"
        
        # action apply
        self.level.player.sprite.get_input(act, space)
        check_death, check_win, check_coin, check_enemy, kill_enemy = self.run()
        
        # get next display after display update
        # self.pygame_update()
        next_obs = self.get_display_img()
        
        # get player distance from goal
        cur_dist = self.level.get_player_from_goal()
        
        # set reward & done
        if check_death or self.cur_health <= 0 or check_win:
            done = True
        else:
            done = False
        
        reward = 0
        if check_win:
            reward = 10
        if check_coin or kill_enemy:
            reward = 5
        if cur_dist < self.goal_dist:
            reward += 1
        else:
            reward -= 1
            
        if check_death:
            reward = -10
        if check_enemy:
            reward = -5
        
        # set cur dist 
        self.goal_dist = cur_dist
        
        return next_obs, reward, done
        
    
    def pygame_update(self):
        pygame.display.update()
        # self.clock.tick(30)
    
    def test_vidoe(self):
        
        self.reset()
        
        while True:
            for event in pygame.event.get():
      
                ### human-play
                keys = pygame.key.get_pressed()

                if keys[pygame.K_RIGHT]:
                    act = "right"
                elif keys[pygame.K_LEFT]:
                    act = "left"
                else:
                    act = "stop"
                space = keys[pygame.K_SPACE]
                self.level.player.sprite.get_input(act, space)
        
                # print(game.coins, game.level.get_player_from_goal())
                        
                if event.type == pygame.QUIT:
        
                    pygame.quit()
                    sys.exit()
            
            # self.screen.fill('grey')
            check_death, check_win, check_coin, check_enemy, kill_enemy = self.run()
        
            if check_coin:
                print("add coin")
            if check_enemy and not kill_enemy:
                print("damage from enemy")
            if check_enemy and kill_enemy:
                print("kill enemy")
            if check_win:
                print("Win!!")
            if check_death or self.cur_health <= 0:
                print("Died")
                self.reset()

            pygame.display.update()
            self.clock.tick(60)
    

# ### for debugging
# pygame.init()
# screen = pygame.display.set_mode((1200, 700), flags=pygame.SHOWN) # flags=pygame.HIDDEN pygame.SHOWN
# env = CustomEnvironment(screen, size_rate=0.2, num_stack=4)

# env.test_vidoe()

# # # idx = 0
# obs = env.reset()
# print(obs.shape)
# obs, _, _ = env.step(4)
# obs, _, _ = env.step(4)

# 이미지를 윈도우에 표시합니다.
# cv2.imshow('Image', obs[-1])

# # 사용자가 키보드의 아무 키나 누를 때까지 대기합니다.
# cv2.waitKey(0)

# # 모든 윈도우를 닫습니다.
# cv2.destroyAllWindows()

# # for _ in range(10):
# #     obs, _, _ = env.step(0,0)

# # # print(obs.shape)

# cv2.imwrite("image.jpg", obs[-1])
    
# while True:
#     env.step(1,1)
    
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             sys.exit()