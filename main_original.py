import pygame, sys
from pygame.locals import *
import math
import random
import os
import json
import numpy
from datetime import datetime

import dispUtils
import dataUtils
import AI
import ObsNRewards

# USER SETTINGS
#Time settings
updatesPerSecond = 1
realTimeFactor = 15

# Multiagent Setting
numberOfAgents = 1

#Spawn Boxs?
spawnBox = False
numberOfBoxes = 1

map_type = ''


# AI Settings
# AImode
# 0: Fresh training
# 1: Continue training from resume_epoch
# 2: Start fresh training but load weights from resume_training_name
# 3: Run AI

AImode = 0
training_name = '2D_3AgentProximity'
resume_epoch = '0' # change to epoch to continue from
resume_training_name = '2D_threeagent_dqn'
saveRate = 10000 #Saves the weights and parameters every x epochs

# Performance Check?
performaceCheck = True
numberOfEpochsToTest = 10
testingInterval = 200
startTesting = 1000

##END OF USER SETTINGS

# Timing how long the training took
startTime = datetime.now()

# set up pygame
pygame.init()
mainClock = pygame.time.Clock()

# set up the window
WINDOWWIDTH = 1100
WINDOWHEIGHT = 1100
windowSurface = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
pygame.display.set_caption('2D Training')

#Generate Map
map_data = dispUtils.mapGeneration(windowSurface,map_type)
windowSurface.fill(map_data[1])
dispUtils.drawObstacles(windowSurface, map_data[2], map_data[0])

# movement Parameters
forawardSpeed = {'linearSpeed': 100/updatesPerSecond, 'angularSpeed': 0/updatesPerSecond}
leftSpeed = {'linearSpeed': 10/updatesPerSecond, 'angularSpeed': 0.6/updatesPerSecond}
rightSpeed = {'linearSpeed': 10/updatesPerSecond, 'angularSpeed': -0.6/updatesPerSecond}
stopSpeed = {'linearSpeed': 0, 'angularSpeed': 0}

# spawn walls and other obstacles
BOXSIZE = WINDOWWIDTH - 200
WALLWIDTH = 50
obstacles = dispUtils.createWalls(windowSurface,BOXSIZE,WALLWIDTH)
dispUtils.drawObstacles(windowSurface, [0,0,0], obstacles)

# box
boxX = int(random.uniform(200, 800))
boxY = int(random.uniform(200, 800))

blacklist = []
while not dispUtils.noCollideSpawnCheck(boxX,boxY,blacklist,200):
    boxX = int(random.uniform(200, 800))
    boxY = int(random.uniform(200, 800))

box = {'position':[boxX,boxY], 'width':80, 'height': 120}
if spawnBox:
    obstacles.append(dispUtils.centreRecttoPoly(box))

def redrawBox():
    boxX = int(random.uniform(200, 800))
    boxY = int(random.uniform(200, 800))
    blacklist = []
    blacklist.extend(robots)
    blacklist.extend(goals)
    while not dispUtils.noCollideSpawnCheck(boxX,boxY,blacklist,200):
        boxX = int(random.uniform(200, 800))
        boxY = int(random.uniform(200, 800))

    box['position'] = [boxX,boxY]
    obstacles[-1] = dispUtils.centreRecttoPoly(box)



# create the AI
path = os.path.dirname(os.path.abspath(__file__)) + '/training_results/' + training_name + '_ep'
reward_file = os.path.dirname(os.path.abspath(__file__)) + '/training_results/' + training_name + '_reward'
performacneReward_file = os.path.dirname(os.path.abspath(__file__)) + '/training_results/' + training_name + '_Performance'

if AImode == 0:
    #Fresh training
    resume_epoch = '0' #freshhhh
    resume_path = os.path.dirname(os.path.abspath(__file__)) + '/training_results/dqn_ep0'
    params_json  = resume_path + '.json'

    dataUtils.create_csv(reward_file)
    if performaceCheck:
        dataUtils.create_csv(performacneReward_file)

    AIparams = dataUtils.loadAIParams(params_json)
    epochs, steps, updateTargetNetwork, explorationRate, epsilon_decay, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_inputs, network_outputs, network_structure, current_epoch = dataUtils.setAIParams(AIparams)
    # network_inputs = network_inputs + (3 - 1)*4
    network_inputs = network_inputs
elif AImode == 1:
    #continue training
    resume_path = path + resume_epoch
    weights_path = resume_path + '.h5'
    params_json  = resume_path + '.json'

    AIparams = dataUtils.loadAIParams(params_json)
    epochs, steps, updateTargetNetwork, explorationRate, epsilon_decay, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_inputs, network_outputs, network_structure, current_epoch = dataUtils.setAIParams(AIparams)

elif AImode == 2:
    #fresh training but with old weights
    params_json  = os.path.dirname(os.path.abspath(__file__)) + '/training_results/dqn_ep0.json'
    weights_path = os.path.dirname(os.path.abspath(__file__)) + '/training_results/' + resume_training_name + '_ep' + resume_epoch + '.h5'

    dataUtils.create_csv(reward_file)
    if performaceCheck:
        dataUtils.create_csv(performacneReward_file)
    AIparams = dataUtils.loadAIParams(params_json)
    epochs, steps, updateTargetNetwork, explorationRate, epsilon_decay, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_inputs, network_outputs, network_structure, current_epoch = dataUtils.setAIParams(AIparams)
    network_inputs = network_inputs + (3 - 1)*4

else:
    #start running
    resume_path = path + resume_epoch
    weights_path = resume_path + '.h5'
    params_json  = resume_path + '.json'

    AIparams = dataUtils.loadAIParams(params_json)
    epochs, steps, updateTargetNetwork, explorationRate, epsilon_decay, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_inputs, network_outputs, network_structure, current_epoch = dataUtils.setAIParams(AIparams)
    explorationRate = 0.01 #Since we dont want any more learning

brain = AI.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
brain.initNetworks(network_structure)


# create the training variables
if not AImode == 0:
    brain.loadWeights(weights_path)
    if not AImode == 2:
        epoch = int(resume_epoch)
    else:
        epoch = 0
else:
    epoch = 0

episode_steps = 0
stepCounter = 0
cumulated_reward = 0
highest_reward = 0
done = False

# Reward Function variablesm
goal_Distance = dataUtils.listOfSize(numberOfAgents,10000)
goal_ang = dataUtils.listOfSize(numberOfAgents,180)
previousVisibility = dataUtils.listOfSize(numberOfAgents,False)

# MultiAgent variables
observation = dataUtils.listOfSize(numberOfAgents,0)
reward = dataUtils.listOfSize(numberOfAgents,0)
action = dataUtils.listOfSize(numberOfAgents,0)

# Spawn All Agents and Goals
# Agent colours
agentColours = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255]]

robots = []
goals = []
for i in range(numberOfAgents):
    # create agent
    robots.append({'position':[100,100], 'size':25 , 'dir':0, 'linearSpeed': 50, 'angularSpeed': 0.1})
    dispUtils.noCollideSpawn(windowSurface,robots[-1],robots[:-1],200)

    # spawn goalpoint
    blacklist = []
    blacklist.extend(robots)
    blacklist.extend(goals)
    goals.append({'position':[100,100], 'size':100})
    dispUtils.noCollideSpawn(windowSurface,goals[-1],blacklist,200)

# PerformanceCheck variables
PerformanceCheckEpochCounter = 0
performanceReward = 0
inPerformanceCheck = False
prePerformanceCheckExplorationRate = 0
checkComplete = False

# run the game loop
while True:
    #Fill Screen with background colour
    windowSurface.fill(map_data[1])

    # check for the QUIT event
    for event in pygame.event.get():
        if event.type == QUIT:
            print(highest_reward)
            print(datetime.now() - startTime)
            dataUtils.plotReward(reward_file,'Culmuilated Reward Over Training')

            if performaceCheck:
                dataUtils.plotReward(performacneReward_file,'Culmuilated Reward Each Performance Check')

            dataUtils.showPlots(False)
            pygame.quit()
            sys.exit()

    #Quit sim if number of epochs is reached in training mode
    if epoch == epochs and AImode != 3:
        print(highest_reward)
        print(datetime.now() - startTime)
        dataUtils.plotReward(reward_file,'Culmuilated Reward Over Training')

        if performaceCheck:
            dataUtils.plotReward(performacneReward_file,'Culmuilated Reward Each Performance Check')

        dataUtils.showPlots(False)
        #Save the weights at the end of the sim
        brain.saveModel(path+str(epoch)+'.h5')
        parameter_keys = ['epochs','steps','updateTargetNetwork','explorationRate','epsilon_decay','minibatch_size','learnStart','learningRate','discountFactor','memorySize','network_inputs','network_outputs','network_structure','current_epoch']
        parameter_values = [epochs, steps, updateTargetNetwork, explorationRate,epsilon_decay, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_inputs, network_outputs, network_structure, epoch]
        parameter_dictionary = dict(zip(parameter_keys, parameter_values))
        with open(path+str(epoch)+'.json', 'w') as outfile:
            json.dump(parameter_dictionary, outfile)
        pygame.quit()
        sys.exit()

    # Draw all the goals and obstacles
    for g in range(len(goals)):
        dispUtils.drawGoal(windowSurface,goals[g],agentColours[g%len(agentColours)])
    dispUtils.drawObstacles(windowSurface, [0,0,0], obstacles)


    dispUtils.drawObstacles(windowSurface, map_data[2], map_data[0])


    # Simulate Each Agent
    for i in range(numberOfAgents):

        #Generate the laser scan data for each agent and check for any collisions
        laserScanData = dispUtils.laserScan(windowSurface, [100,100,100], robots[i], agentColours, 10, [-math.pi/2,math.pi/2], 500,54)
        if dispUtils.checkCollision(laserScanData,55) == True:
            done = True
        #print(laserScanData)

        # Create list of other Agents
        others = robots[:i]
        others.extend(robots[(i+1):])

        # Check for collisions with other robots
        for otherRobot in others:
            if dispUtils.getInterAgentDistace(robots[i],otherRobot)< robots[i]['size']*2:
                done = True

        # Draw the robot and create the new observation list
        dispUtils.drawAgent(windowSurface,agentColours[i%len(agentColours)],robots[i])
        newObservation = ObsNRewards.getObsevation(windowSurface,laserScanData,robots[i],goals[i],others)
        #print(newObservation)


        reward[i], goal_ang[i], goal_Distance[i] = ObsNRewards.getReward(windowSurface, robots[i], goals[i], others, newObservation, action[i], goal_ang[i], goal_Distance[i])
        cumulated_reward += reward[i]
        if cumulated_reward > highest_reward:
            highest_reward = cumulated_reward

        # add this memory to the AI, this only happens after the AI has made a decision and observed the result
        if stepCounter > 0 and AImode != 3 and not inPerformanceCheck and episode_steps>0:
            brain.addMemory(numpy.array(observation[i]), action[i], reward[i], numpy.array(newObservation), isFinal)

        # Check if the previous action has lead to an ending state ie collison or goal
        isFinal = False
        if newObservation[-1] == 1:
            isFinal = True
        else:
            isFinal = False

        if episode_steps == steps-1:
            done = True

        isFinal = isFinal or done

        # Begin learning from the collected memories once a sufficient number have been collected
        if stepCounter >= learnStart and stepCounter%minibatch_size == 0 and AImode != 3 and not inPerformanceCheck:
            #print('learning')
            if stepCounter <= updateTargetNetwork:
                brain.learnOnMiniBatch(minibatch_size, False)
            else :
                brain.learnOnMiniBatch(minibatch_size, True)

        #AI makes decision
        qValues = brain.getQValues(numpy.array(newObservation))
        action[i] = brain.selectAction(qValues, explorationRate)
        # action = int(random.uniform(0,3))

        # Process action
        if action[i] == 0:  # FORWARD
            robots[i]['linearSpeed'] = forawardSpeed['linearSpeed']
            robots[i]['angularSpeed'] = forawardSpeed['angularSpeed']
        elif action[i] == 1:  # LEFT
            robots[i]['linearSpeed'] = leftSpeed['linearSpeed']
            robots[i]['angularSpeed'] = leftSpeed['angularSpeed']
        elif action[i] == 2:  # RIGHT
            robots[i]['linearSpeed'] = rightSpeed['linearSpeed']
            robots[i]['angularSpeed'] = rightSpeed['angularSpeed']
        elif action[i] == 3:  # STOP
            robots[i]['linearSpeed'] = stopSpeed['linearSpeed']
            robots[i]['angularSpeed'] = stopSpeed['angularSpeed']
        # Observations are now old
        observation[i] = newObservation


    if done == True:
        if inPerformanceCheck or epoch%saveRate == 0:
            map_data = dispUtils.mapGeneration(windowSurface,map_type)
            windowSurface.fill(map_data[1])
            dispUtils.drawObstacles(windowSurface, [0,0,0], obstacles)
            dispUtils.drawObstacles(windowSurface, map_data[2], map_data[0])
        # If we're spawning boxs then reset that too
        if spawnBox:
            redrawBox()

        # This is for when the robots have collided or taken too many steps
        for i in range(numberOfAgents):
            # reset start points
            dispUtils.noCollideSpawn(windowSurface,robots[i],robots[:i],200)

            # reset goals goalpoint
            blacklist = []
            blacklist.extend(robots)
            blacklist.extend(goals[:i])
            dispUtils.noCollideSpawn(windowSurface,goals[i],blacklist,200)

        # If we're training then save the reward from that epoch
        if AImode != 3 and not inPerformanceCheck:
            dataUtils.save_rewards(reward_file,epoch,cumulated_reward,stepCounter)

        # Save the model if we've hit saveRate number of epochs
        if (epoch)%saveRate==0 and epoch>=100 and AImode != 3 and not inPerformanceCheck:
            #save model weights and monitoring data every 100 epochs.
            brain.saveModel(path+str(epoch)+'.h5')
            parameter_keys = ['epochs','steps','updateTargetNetwork','explorationRate','epsilon_decay','minibatch_size','learnStart','learningRate','discountFactor','memorySize','network_inputs','network_outputs','network_structure','current_epoch']
            parameter_values = [epochs, steps, updateTargetNetwork, explorationRate,epsilon_decay, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_inputs, network_outputs, network_structure, epoch]
            parameter_dictionary = dict(zip(parameter_keys, parameter_values))
            with open(path+str(epoch)+'.json', 'w') as outfile:
                json.dump(parameter_dictionary, outfile)


        if inPerformanceCheck:
            if PerformanceCheckEpochCounter >= numberOfEpochsToTest-1:
                performanceResult = performanceReward/numberOfEpochsToTest
                print('Performance Check Complete. Reward Obtained: ' + str(performanceResult))
                inPerformanceCheck = False
                dataUtils.save_rewards(performacneReward_file,epoch,performanceResult,stepCounter)
                explorationRate = prePerformanceCheckExplorationRate
                epoch += 1
                checkComplete = True
            PerformanceCheckEpochCounter += 1
            performanceReward += cumulated_reward

        if performaceCheck and not inPerformanceCheck and epoch%testingInterval == 0 and AImode != 3 and epoch>=startTesting:
            print('Starting Performance Check. Epoch: ' + str(epoch))
            inPerformanceCheck = True
            PerformanceCheckEpochCounter = 0
            performanceReward = 0
            prePerformanceCheckExplorationRate = explorationRate
            explorationRate  = 0.05
        else:
            if not inPerformanceCheck:
                if checkComplete:
                    checkComplete = False
                else:
                    epoch += 1
                stepCounter += 1

        episode_steps = 0
        done = False
        cumulated_reward = 0

        #Decrease exploration
        if explorationRate > 0.05:
            explorationRate *= epsilon_decay
            explorationRate = max(0.05, explorationRate)
            if epoch%200==0:
                print('Epoch: ' + str(epoch) + ', Exploration Rate: ' + str(explorationRate))
    else:
        for i in range(numberOfAgents):
            # Reset that agents goal if they reached it
            if observation[i][-1] == 1:
                blacklist = []
                blacklist.extend(robots)
                blacklist.extend(goals[:i])
                blacklist.extend(goals[(i+1):])
                dispUtils.noCollideSpawn(windowSurface,goals[i],blacklist,200)

            # Update agent's position
            dispUtils.updateAgent(robots[i])

        if not inPerformanceCheck:
            stepCounter += 1

        episode_steps += 1

    if stepCounter % updateTargetNetwork == 0 and AImode != 3 and not inPerformanceCheck:
        brain.updateTargetNetwork()
        print ("Step " + str(stepCounter) + ": updating target network")

    pygame.display.update()
    mainClock.tick(updatesPerSecond * realTimeFactor)
