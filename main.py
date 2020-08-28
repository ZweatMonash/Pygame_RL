import os
import AI
import dataUtils
from environment import Environment
from agent import Agent
import pygame
import numpy as np
import json

#Time
mainClock = pygame.time.Clock()
updatesPerSecond = 1
realTimeFactor =  50
import time

AImode = 1
performaceCheck = False
numberOfEpochsToTest = 10
testingInterval = 200
startTesting = 1000
saveRate = 10000 #Saves the weights and parameters every x epochs
resume_epoch = '9480' # change to epoch to continue from
resume_training_name = 'Simple_World'
training_name = "Simple_World"

# create the AI
path = os.path.dirname(os.path.abspath(__file__)) + '/training_results/' + training_name + '_ep'
reward_file = os.path.dirname(os.path.abspath(__file__)) + '/training_results/' + training_name + '_reward.csv'
performacneReward_file = os.path.dirname(os.path.abspath(__file__)) + '/training_results/' + training_name + '_Performance.csv'


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



# PerformanceCheck variables
PerformanceCheckEpochCounter = 0
performanceReward = 0
inPerformanceCheck = False
prePerformanceCheckExplorationRate = 0
checkComplete = False

#Environment variables
epoch = 0
highest_reward = 0
env = Environment()
targetStepCounter = 0



while True: 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            dataUtils.plotReward(reward_file,'Culmuilated Reward Over Training')
            if performaceCheck: 
                dataUtils.plotReward(performacneReward_file,'Culmuilated Reward Each Performance Check')
            pygame.quit()

    done = False
    stepCounter = 0
    cumulated_reward = 0
    observation = env.reset()
    # print(observation)
    while not done:
        action = 3
        qValues = brain.getQValues(observation)
        action = brain.selectAction(qValues,explorationRate)
        
        newObservation, reward, done, _ = env.step(action)
        
        cumulated_reward += reward
        if highest_reward < cumulated_reward:
            highest_reward = cumulated_reward

        brain.addMemory(observation, action, reward, newObservation, done)

        if targetStepCounter >= learnStart and targetStepCounter%minibatch_size == 0:
            if targetStepCounter <= updateTargetNetwork:
                brain.learnOnMiniBatch(minibatch_size, False)
            else :
                brain.learnOnMiniBatch(minibatch_size, True)
        
        observation = newObservation #next state

        if targetStepCounter % updateTargetNetwork == 0:
            targetStepCounter = 0
            brain.updateTargetNetwork()
            print ("updating target network")

        if done == True:
            
             # If we're training then save the reward from that epoch
            if AImode != 3:
                dataUtils.save_rewards(reward_file,epoch,cumulated_reward,stepCounter)

            # Save the model if we've hit saveRate number of epochs
            if epoch%1000 == 0:
                #save model weights and monitoring data every 100 epochs.
                brain.saveModel(path+str(epoch)+'.h5')
                parameter_keys = ['epochs','steps','updateTargetNetwork','explorationRate','epsilon_decay','minibatch_size','learnStart','learningRate','discountFactor','memorySize','network_inputs','network_outputs','network_structure','current_epoch']
                parameter_values = [epochs, steps, updateTargetNetwork, explorationRate,epsilon_decay, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_inputs, network_outputs, network_structure, epoch]
                parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                with open(path+str(epoch)+'.json', 'w') as outfile:
                    json.dump(parameter_dictionary, outfile)

            epoch += 1
            explorationRate *= epsilon_decay
            explorationRate = max(0.05, explorationRate)
            
        stepCounter += 1
        targetStepCounter += 1
        mainClock.tick(updatesPerSecond * realTimeFactor)
    

   


        