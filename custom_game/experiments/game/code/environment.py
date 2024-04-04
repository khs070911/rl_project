import gym
import cv2
import numpy as np
import pygame, sys, random
from settings import screen_height, screen_width
from level import Level
from ui import UI


class CustomEnvironment:
    
    def __init__(self, screen):
        
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
        
        # set obs
        img = self.get_display_img()
        
        # set distance
        self.goal_dist = self.level.get_player_from_goal()
        
        # pygame update
        self.pygame_update()
        
        return img
    
    def get_display_img(self):
        # state
        screen = pygame.display.get_surface()
        img = pygame.surfarray.array3d(screen)
        img = img.transpose([1, 0, 2])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        
        return img
    
    def step(self, act, space):
        """
            act: 0, 1, 2. 0 = stop, 1 = right, 2 = left
            space: 0, 1. if True, player jump
        """
        
        if act == 1:
            act = "right"
        elif act == 2:
            act = "left"
        else:
            act = "stop"
        
        space = space == 1
        
        # action apply
        self.level.player.sprite.get_intput(act, space)
        check_death, check_win, check_coin, check_enemy, kill_enemy = self.run()
        
        # get next display after display update
        self.pygame_update()
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
        self.clock.tick(60)
    
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
            
                    # ### 게임 이미지 출력
                    # screen = pygame.display.get_surface()
                    # img = pygame.surfarray.array3d(screen)
                    # img = img.transpose([1, 0, 2])
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # img = cv2.resize(img, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
                    # cv2.imwrite("image.jpg", img)
                    # #######################################
        
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
    

### for debugging
# pygame.init()
# screen = pygame.display.set_mode((screen_width,screen_height), flags=pygame.SHOWN) # flags=pygame.HIDDEN
# env = CustomEnvironment(screen)

# env.test_vidoe()