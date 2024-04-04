import pygame, sys, random
from settings import screen_height, screen_width
from level import Level
from overworld import Overworld
from ui import UI

import numpy as np
import cv2

class Game:
	def __init__(self):

		# game attributes
		self.max_level = 2
		self.max_health = 100
		self.cur_health = 100
		self.coins = 0
		
		# audio 
		self.level_bg_music = pygame.mixer.Sound('../audio/level_music.wav')
		self.overworld_bg_music = pygame.mixer.Sound('../audio/overworld_music.wav')

		# overworld creation
		self.overworld = Overworld(0,self.max_level,screen,self.create_level)
		self.status = 'overworld'
		self.overworld_bg_music.play(loops = -1)

		# user interface 
		self.ui = UI(screen)


	def create_level(self,current_level):
		self.level = Level(current_level,screen,self.create_overworld,self.change_coins,self.change_health)
		self.status = 'level'
		self.overworld_bg_music.stop()
		self.level_bg_music.play(loops = -1)

	def create_overworld(self,current_level,new_max_level):
		if new_max_level > self.max_level:
			self.max_level = new_max_level
		self.overworld = Overworld(current_level,self.max_level,screen,self.create_level)
		self.status = 'overworld'
		self.overworld_bg_music.play(loops = -1)
		self.level_bg_music.stop()

	def change_coins(self,amount):
		self.coins += amount

	def change_health(self,amount):
		self.cur_health += amount

	def check_game_over(self):
		if self.cur_health <= 0:
			self.cur_health = 100
			self.coins = 0
			self.max_level = 0
			self.overworld = Overworld(0,self.max_level,screen,self.create_level)
			self.status = 'overworld'
			self.level_bg_music.stop()
			self.overworld_bg_music.play(loops = -1)

	def run(self):
		if self.status == 'overworld':
			self.overworld.run()
		else:
			self.level.run()
			self.ui.show_health(self.cur_health,self.max_health)
			self.ui.show_coins(self.coins)
			self.check_game_over()

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((screen_width,screen_height), flags=pygame.SHOWN) # flags=pygame.HIDDEN
clock = pygame.time.Clock()
game = Game()
game.create_level(0) ## play stage 고정


while True:
    # ### random play
	# act = random.choice(["right", "left", "stop"])
	# space = random.choice([True, False])
	# game.level.player.sprites()[0].get_input(act, space)
 
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
		game.level.player.sprite.get_input(act, space)
  
		print(game.coins)
				
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
	
	screen.fill('grey')
	game.run()

	pygame.display.update()
	clock.tick(60)