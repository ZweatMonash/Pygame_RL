import dispUtils
import dataUtils
import AI

import pygame
import numpy as np
import gym
import math
from agent import Agent
from pedestrain import Creep

from gym import error, utils
from gym.utils import seeding
from random import random

creeps_enabled = False

class Environment(gym.Env):
    """
    Class as a wrapper for the Simulation2D.
    """ 
    # updatesPerSecond = 10
    # realTimeFactor =  10
    WINDOWWIDTH = 1100
    WINDOWHEIGHT = 1100
    NUMAGENTS = 1
    
    

    def __init__(self):
        """
        Constructor to initialize the environment.
        :param path_to_world: The path to the world which should be selected.
        """

        # set up pygame
        pygame.init()
        pygame.display.set_caption('DQN_pygame')

        self.window_width = self.WINDOWWIDTH
        self.window_height = self.WINDOWHEIGHT
        self.num_agents = self.NUMAGENTS
        self.mainClock = pygame.time.Clock()
        self.windowSurface = pygame.display.set_mode((self.window_width, self.window_height))
        self.map_type = ''
        self.blacklist = []
        self.collisionlist = []

        self.goal_x = 800
        self.goal_y = 800
        self.goal_size = 100

        self.agent = Agent(self.windowSurface ,300, 300, self.goal_x, self.goal_y, radius=0.5, pref_speed=1.0, initial_heading=0.0, id=0)
        if creeps_enabled:        
            self.creep0 = Creep(self.windowSurface,25,(300,800),(1,1),3,self.WINDOWWIDTH,self.WINDOWHEIGHT,random())
            self.creep1 = Creep(self.windowSurface,25,(800,300),(1,-1),3,self.WINDOWWIDTH,self.WINDOWHEIGHT,random())
            self.creep2 = Creep(self.windowSurface,25,(500,500),(-1,1),3,self.WINDOWWIDTH,self.WINDOWHEIGHT,random())
            self.creep3 = Creep(self.windowSurface,25,(200,700),(1,-1),3,self.WINDOWWIDTH,self.WINDOWHEIGHT,random())

        self.max_steps = 2000
        self.curr_step = 0
        self.done = False

    def draw_map(self):
        """
        Create a map using mapGeneration and filling it to window_surface
        """
        map_data = dispUtils.mapGeneration(self.windowSurface,self.map_type)
        self.windowSurface.fill(map_data[1])
        dispUtils.drawObstacles(self.windowSurface, map_data[2], map_data[0])

    def draw_walls(self):
        """
        Spawn walls and other obstacles and draw on window surface
        """
        BOXSIZE = self.window_width - 200
        WALLWIDTH = 50
        obsList = dispUtils.createWalls(self.windowSurface,BOXSIZE,WALLWIDTH)
        for shape in obsList:
            pygame.draw.polygon(self.windowSurface, [0,0,0], shape)


    def draw_goals(self):
        """
        Spawn goal and draw on windowsurface
        """
        goalColor = [255,0,0]
        drawColor = [0,0,0]
        for i in range(3):
            drawColor[0] = max(0,goalColor[0] - 10 - 50*(3-i))
            drawColor[1] = max(0,goalColor[1] - 10 - 50*(3-i))
            drawColor[2] = max(0,goalColor[2] - 10 - 50*(3-i))
            pygame.draw.circle(self.windowSurface, drawColor, (self.goal_x,self.goal_y), self.goal_size/(i+1))


    def update_collisionlist(self):
        self.collisionlist = []
        self.collisionlist.extend(self.creep0.pos)
        self.collisionlist.extend(self.creep1.pos)
        self.collisionlist.extend(self.creep2.pos)
        self.collisionlist.extend(self.creep3.pos)

    def update_blacklist(self,pos_tuple):
        self.blacklist.append(pos_tuple)

    def spawn(self):
        self.windowSurface.fill((255,255,255)) #white screen
        self.draw_map()
        self.draw_walls()
        self.goal_x,self.goal_y = dispUtils.noCollideSpawn(self.windowSurface,self.blacklist,200)
        self.update_blacklist((self.goal_x,self.goal_y))
        self.agent.set_goal(self.goal_x,self.goal_y)
        self.draw_goals()
        self.agent.x,self.agent.y = dispUtils.noCollideSpawn(self.windowSurface,self.blacklist,300)
        self.update_blacklist(( self.agent.x,self.agent.y))
        self.agent.draw()
        if creeps_enabled: 
            self.creep0.draw()
            self.creep1.draw()
            self.creep2.draw()
            self.creep3.draw()

        pygame.display.update()

    def reset_goal(self):
        self.blacklist = []
        self.update_blacklist(( self.agent.x,self.agent.y))
        self.goal_x,self.goal_y = dispUtils.noCollideSpawn(self.windowSurface,self.blacklist,200)
        self.update_blacklist((self.goal_x,self.goal_y))
        self.agent.set_goal(self.goal_x,self.goal_y)
    
    def redraw(self):
        self.windowSurface.fill((255,255,255)) #white screen
        self.draw_map()
        self.draw_walls()
        self.draw_goals()
        self.agent.draw()
        if creeps_enabled: 
            self.creep0.draw()
            self.creep1.draw()
            self.creep2.draw()
            self.creep3.draw()

        pygame.display.update()

    def get_reward(self,observation,action):
        
        reward = 0
        if action == 0:  # FORWARD
            reward += 0
        elif action == 1:  # LEFT
            reward += 0
        elif action == 2:  # RIGHT
            reward += 0
        elif action == 3:  # STOP
            reward += -2
        # print("now" + str(self.agent.prev_relAngleDif))            
        # print("now"/ + str(self.agent.relAngleDif))
        if(self.agent.prev_relAngleDif> self.agent.relAngleDif or self.agent.relAngleDif < 5):
            reward += 2
     
        if self.agent.prev_goal_dis > self.agent.goal_dis :
            reward += 5
        else:
            reward -= 5

        if self.done:
            reward += -1000
        elif self.agent.goal_dis < 0.5:
            reward += 300

        return reward


    def step(self,action):
        if creeps_enabled: 
            self.creep0.update() #update postiion of the creep
            self.creep1.update()
            self.creep2.update()
            self.creep3.update()
        self.agent.get_actions(action) #map action index to velocity
        self.agent.update_agent() #update the state of the agent based on action taken
        
        if self.agent.is_at_goal() :
            self.reset_goal()
        self.redraw()
        observation = self.agent.get_observations(self.windowSurface)
        self.is_done(observation)
        reward = self.get_reward(observation,action)
        info = None
        return observation, reward, self.done, info


    def is_done(self,observation):
        if dispUtils.checkCollision(observation[0:10],55) == True : #or self.curr_step >= self.max_steps
            self.done = True

    def teleop(self):

        self.keys = pygame.key.get_pressed()

        if self.keys[pygame.K_UP]:
            action = 0
        elif self.keys[pygame.K_LEFT]:
            action = 1
        elif self.keys[pygame.K_RIGHT]:
            action = 2
        else:
            action = 3

        return action

    def reset(self):
        self.blacklist = []
        self.done = False
        self.curr_step = 0
        self.agent.theta = 0
        self.spawn()
        observation = self.agent.get_observations(self.windowSurface)
        return observation




    

