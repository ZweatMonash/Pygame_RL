import math

import dispUtils


# The State the robot will observe
def getObsevation(surface,LS,agent,target,otherAgents):
    #[10LS, Distance_to_goal, 2* other_agents(distance_to_other_agent, angle_to_that, agnet_speed, direction_facing), rel_heading_goal_rel_robot[0,2pi], done]
    newObs = []
    newObs.extend(LS) #Add the laserscan data
    newObs.append(round(dispUtils.getInterAgentDistace(agent,target)/100,2)) #Distance to goal
    
    # #The robot knows if it can see its goal
    # if dispUtils.lineOfSight(surface,agent,target,54):
    #     visibleGoal = 1
    # else:
    #     visibleGoal = 0
    #
    # newObs.append(visibleGoal)

    # The angle its facing
    if agent['dir'] < 0:
        obsAngle = agent['dir'] + 2*math.pi
    else:
        obsAngle = agent['dir']



    # closetAgentsID = [0,0]
    # closestAgentsDist = [1000000,1000000]
    # IDcounter = 0
    # for otherAgent in otherAgents:
    #     if dispUtils.getInterAgentTheta(agent,otherAgent) < math.pi/2 and dispUtils.getInterAgentTheta(agent,otherAgent) > -math.pi/2:
    #         if dispUtils.getInterAgentDistace(agent,otherAgent)<=closestAgentsDist[0]:
    #             closestAgentsDist[1] = closestAgentsDist[0]
    #             closetAgentsID[1] = closetAgentsID[0]
    #             closestAgentsDist[0] = dispUtils.getInterAgentDistace(agent,otherAgent)
    #             closetAgentsID[0] = IDcounter
    #         elif dispUtils.getInterAgentDistace(agent,otherAgent)<=closestAgentsDist[1]:
    #             closestAgentsDist[1] = dispUtils.getInterAgentDistace(agent,otherAgent)
    #             closetAgentsID[1] = IDcounter
    #     IDcounter += 1

    # closestAgents = [otherAgents[closetAgentsID[0]],otherAgents[closetAgentsID[1]]]


    # for otherAgent in closestAgents:
    #     #distance to nearest 2 other robot
    #     if dispUtils.getInterAgentDistace(agent,otherAgent) <= 500:
    #         newObs.append(round(dispUtils.getInterAgentDistace(agent,otherAgent)/100,2))
    #         # The angle between the robot and the other robot relative to the front of robot
    #         if dispUtils.getInterAgentTheta(agent,otherAgent) < 0:
    #             obsRelAngle = dispUtils.getInterAgentTheta(agent,otherAgent) + 2*math.pi
    #         else:
    #             obsRelAngle = dispUtils.getInterAgentTheta(agent,otherAgent)

    #         relAngleDif = obsRelAngle - obsAngle

    #         if relAngleDif > math.pi:
    #             relAngleDif = relAngleDif-2*math.pi
    #         elif relAngleDif < -math.pi:
    #             relAngleDif = relAngleDif+2*math.pi

    #         newObs.append(round(abs(math.degrees(relAngleDif)),1))

    #         #speed of other robot
    #         newObs.append(round(float(otherAgent['linearSpeed'])/100,1))
    #         #orientation of other robot relative to front of robot
    #         if otherAgent['dir'] < 0:
    #             obsRelAngle = otherAgent['dir'] + 2*math.pi
    #         else:
    #             obsRelAngle = otherAgent['dir']

    #         relAngleDif = obsRelAngle - obsAngle

    #         if relAngleDif > math.pi:
    #             relAngleDif = relAngleDif-2*math.pi
    #         elif relAngleDif < -math.pi:
    #             relAngleDif = relAngleDif+2*math.pi

    #         newObs.append(round(abs(math.degrees(relAngleDif)),1))

    #     else:
    #         newObs.append(round(500/100,2)) #5m laser range
    #         newObs.append(round(abs(math.degrees(0)),1))
    #         newObs.append(0)
    #         newObs.append(round(abs(math.degrees(0)),1))


    #newObs.append(round(math.degrees(obsAngle),1))

    # The angle between the robot and the goal relative to the world
    if dispUtils.getInterAgentTheta(agent,target) < 0:
        obsRelAngle = dispUtils.getInterAgentTheta(agent,target) + 2*math.pi
    else:
        obsRelAngle = dispUtils.getInterAgentTheta(agent,target)
    #newObs.append(round(abs(math.degrees(obsRelAngle)),1))

    # The angle between the robot and the goal relative to the front of the robot
    relAngleDif = obsRelAngle - obsAngle

    if relAngleDif > math.pi:
        relAngleDif = relAngleDif-2*math.pi
    elif relAngleDif < -math.pi:
        relAngleDif = relAngleDif+2*math.pi

    newObs.append(round(abs(math.degrees(relAngleDif)),1))

    # If the robot is at the goal or not
    if dispUtils.getInterAgentDistace(agent,target) < 50:
        arrived = 1
    else:
        arrived = 0

    newObs.append(arrived)
    return newObs

def getReward(surface, agent, target, otherAgents, observation, action, prevtargetAng, prevtargetDist):
    #REWARD FUNCTION
    # Determine the reward for the previous action taken
    # Reward for movement
    reward = -2
    if action == 0:  # FORWARD
        reward += 0
    elif action == 1:  # LEFT
        reward += 0
    elif action == 2:  # RIGHT
        reward += 0
    elif action == 3:  # STOP
        if min(observation[3:6])<1:
            reward -= 0
        else:
            reward -= 100

    # Reward for facing the goal or moving toward facing the goal
    if observation[-2] < 2 or prevtargetAng > observation[-2]:
        reward+=2
    prevtargetAng = observation[-2]

    # Agent to Agent reward
    # Dont get too close
    for oi in range(len(otherAgents)):
        if dispUtils.getInterAgentDistace(agent,otherAgents[oi]) < 200:
            reward-=5
            if dispUtils.getInterAgentDistace(agent,otherAgents[oi]) < 100:
                reward-= 5 + (100-dispUtils.getInterAgentDistace(agent,otherAgents[oi])/100)*5

        # Punish collision
        if dispUtils.getInterAgentDistace(agent,otherAgents[oi])< agent['size']*2:
            reward-=1000

    # Goal and Obstacle related Reward
    if dispUtils.getInterAgentDistace(agent,target) < 50:
        reward += 300 #at Goal
    elif dispUtils.getInterAgentDistace(agent,target) < prevtargetDist - 10:
        reward += 5 # Closer to goal
    elif dispUtils.checkCollision(observation[0:9],55) == True:
        reward -= 1000 #collide with obstacle
    else:
        reward -= 5

    prevtargetDist = dispUtils.getInterAgentDistace(agent,target)

    # Punnish driving foward close to walls
    # for scan in observation[4:5]:
    #     if scan <= 1:
    #         reward-= 5 + ((1-scan)/1)*5

    return reward, prevtargetAng, prevtargetDist
