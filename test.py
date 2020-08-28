#!/usr/bin/env python

import pygame, sys, math, random
from pygame.locals import *
from datetime import datetime
import rvo2

import dispUtils
import dataUtils

startTime = datetime.now()

# set up pygame
pygame.init()
mainClock = pygame.time.Clock()

# set up the window
WINDOWWIDTH = 1000
WINDOWHEIGHT = 1000
windowSurface = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
pygame.display.set_caption('2D Training')
windowSurface.fill([255, 255, 255])

# spawn walls and other obstacles
BOXSIZE = 800
WALLWIDTH = 50
obstacles = dispUtils.createWalls(windowSurface,BOXSIZE,WALLWIDTH)

# box
boxX = int(random.uniform(200, 800))
boxY = int(random.uniform(200, 800))




########################################################################################################################################################################

#  timestep, neighbourDist, maxNeigh, timeHorizon, timeHorizion obstac, radius, maxspeed, velocity(vector2)
numberOfAgents = 5

sim = rvo2.PyRVOSimulator(1/60., 1.5, numberOfAgents , 1.5, 2, 0.4, 2)
# Pass either just the position (the other parameters then use
# the default values passed to the PyRVOSimulator constructor),
# or pass all available parameters.
a0 = sim.addAgent((0, 0))
a1 = sim.addAgent((1, 0))
a2 = sim.addAgent((1, 1))
a3 = sim.addAgent((0, 1), 1.5, 5, 1.5, 2, 0.4, 2, (0, 0))

# Obstacles are also supported.
o1 = sim.addObstacle([(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1)])
sim.processObstacles()

sim.setAgentPrefVelocity(a0, (1, 1))
sim.setAgentPrefVelocity(a1, (-1, 1))
sim.setAgentPrefVelocity(a2, (-1, -1))
sim.setAgentPrefVelocity(a3, (1, -1))

########################################################################################################################################################################

# Spawn All Agents and Goals
robots = []
goals = []
for i in range(numberOfAgents): 
    # create agent
    agentX = WINDOWWIDTH/2 + math.cos(math.pi - (float(i)*2*math.pi)/float(numberOfAgents))*200
    agentY = WINDOWHEIGHT/2 + math.sin(math.pi - (float(i)*2*math.pi)/float(numberOfAgents))*200
    robots.append({'position':[int(agentX),int(agentY)], 'size':25 , 'dir':0, 'linearSpeed': 50, 'angularSpeed': 0.1})

    # spawn goalpoint
    blacklist = []
    blacklist.extend(robots)
    blacklist.extend(goals)
    goals.append({'position':[100,100], 'size':100})
    dispUtils.noCollideSpawn(goals[-1],blacklist,200)

done = False

# run the game loop

while(True):
    sim.doStep()

    
    #Fill Screen with white background
    windowSurface.fill([255, 255, 255])

    # check for the QUIT event
    for event in pygame.event.get():
        if event.type == QUIT:
                pygame.quit()
                sys.exit()

    dispUtils.drawAgent(windowSurface,[255,0,0],robots[i])


    # Draw all the goals and obstacles
    for goal in goals:
        dispUtils.drawGoal(windowSurface,goal)
    dispUtils.drawObstacles(windowSurface, [0,0,0], obstacles)

    for i in range(numberOfAgents):
         dispUtils.drawAgent(windowSurface,[255,0,0],robots[i])
        


    if done == False:
        dispUtils.updateAgent(robots[i])



        