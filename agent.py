import numpy as np
import operator
import pygame
import math
import dispUtils

class Agent():

    # movement Parameters of Agent
    updatesPerSecond = 1   # steps per sec overserbing
    realTimeFactor =  50   # related to the clock /speeding up for training
    forwardSpeed = {'linearSpeed': 50/updatesPerSecond, 'angularSpeed': 0/updatesPerSecond}
    leftSpeed = {'linearSpeed':10/updatesPerSecond, 'angularSpeed': 0.6/updatesPerSecond}
    rightSpeed = {'linearSpeed': 10/updatesPerSecond, 'angularSpeed': -0.6/updatesPerSecond}
    stopSpeed = {'linearSpeed': 0, 'angularSpeed': 0}
    agentColours = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255]]

    def __init__(self, windowSurface ,start_x, start_y, goal_x, goal_y, radius=0.5, pref_speed=1.0, initial_heading=0.0, id=0):

        self.windowSurface = windowSurface
        self.policy_type = "DQN"
        self.x = start_x
        self.y = start_y
        
        self.size = 25 #pygame pixel in radius
        self.theta = initial_heading #[ - pi , pi]
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.pref_speed = pref_speed
        self.id = id #0 means host, >0 means other agent/pedestrains
        self.goal_dis = None
        self.relAngleDif = None
        self.prev_goal_dis = 1000.0 #initilizing a big number
        self.prev_relAngleDif = 180 #initilizing a big number
        self.prev_x = None
        self.prev_y = None
        self.prev_theta  = None
        
        self.linear_v = 0.0
        self.angular_v = 0.0

    def get_actions(self,action):

        if action == 0:  # FORWARD
            self.linear_v = self.forwardSpeed['linearSpeed']
            self.angular_v = self.forwardSpeed['angularSpeed']
        elif action == 1:  # LEFT
            self.linear_v = self.leftSpeed['linearSpeed']
            self.angular_v = self.leftSpeed['angularSpeed']
        elif action == 2:  # RIGHT
            self.linear_v = self.rightSpeed['linearSpeed']
            self.angular_v = self.rightSpeed['angularSpeed']
        # elif action == 3:  # STOP
        #     self.linear_v = self.stopSpeed['linearSpeed']
        #     self.angular_v = self.stopSpeed['angularSpeed']

    def update_agent(self):
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_theta = self.theta
        self.theta = self.theta + self.angular_v #new theta
        if self.theta > math.pi:
            self.theta = self.theta - 2*math.pi 
        elif self.theta < -math.pi:
            self.theta = self.theta + 2*math.pi
        # print("theta in update_agent " +str(round(self.theta*180/math.pi)))
        self.x = int(self.x + self.linear_v * math.cos(self.theta)) #new X
        self.y = int(self.y + self.linear_v * math.sin(self.theta)) #new Y


    def get_laserScan(self):
        
        laserScanData = []
        divisions = 10 # Number of laser scans
        scanColour = [0,0,0] #scan for black object
        AngRange = [-math.pi/2,math.pi/2] # angular range of scan in rad
        scanDis = 500 # Maximum scan distance in pix
        scanMinRange = 54 #Minimum scan distance (robot radius - 50)
        for i in range(divisions+1):
            scanAng = AngRange[0] + self.theta + i*(AngRange[1]-AngRange[0])/divisions
            scanEndX = int(self.x + (scanMinRange * math.cos(scanAng)))
            scanEndY = int(self.y + (scanMinRange * math.sin(scanAng)))

            #Check for collision
            for scanReach in range((scanMinRange/10) +1,(scanDis/10)+1):
                if scanEndX < self.windowSurface.get_width() and scanEndX > 0 and scanEndY < self.windowSurface.get_height() and scanEndY > 0:
                    collision = False
                    for colour in self.agentColours:
                        if dispUtils.isColor(self.windowSurface.get_at((scanEndX,scanEndY))[:3],colour):
                            collision = True
                            break
                    if dispUtils.isColor(self.windowSurface.get_at((scanEndX,scanEndY))[:3],[0,0,0]) or collision:
                        break
                    else:
                        scanEndX = int(self.x + (float(scanReach)*10 * math.cos(scanAng)))
                        scanEndY = int(self.y + (float(scanReach)*10 * math.sin(scanAng)))
                else:
                    break

            pygame.draw.line(self.windowSurface, scanColour,[self.x,self.y],[scanEndX,scanEndY])
            laserScanData.append(round((math.hypot(scanEndX - self.x, scanEndY - self.y))/100,1))
        pygame.display.update()
        return laserScanData

    def is_at_goal(self):
        return True if (self.goal_dis < 0.5) else False

    def set_goal(self,x,y):
        self.goal_x = x
        self.goal_y = y

    def get_observations(self,surface):
        #[11 LS, goal dis, yaw, rel_ang_to_goal, angle_diff, arrived] #16 states
        self.prev_relAngleDif = self.relAngleDif
        self.prev_goal_dis = self.goal_dis
        self.windowSurface = surface
        newObs = []
        arrived = False
        laserScan = self.get_laserScan() #normalised by 100
        dis_goal = round(dispUtils.getInterAgentDistace([self.x,self.y],[self.goal_x,self.goal_y])/100,1) #normalsied by 100
        obsAngle = round(self.theta,2) #[-180, 180] robot's yaw
        obsRelAngle = round(dispUtils.getInterAgentTheta([self.x,self.y],[self.goal_x,self.goal_y]),2) # [-180,180] Ang betwn robot and goal relative to world

        obsAngle = round(obsAngle * 180/math.pi) #[-180,180]
        obsRelAngle = round(obsRelAngle * 180/math.pi) #[-180,180]
        relAngleDif = abs(obsRelAngle - obsAngle) #[0,180]

        if relAngleDif <= 180:
                relAngleDif = relAngleDif
        else:
            relAngleDif = 360 - relAngleDif

        if dis_goal < 0.5:
            arrived = True

        newObs.extend(laserScan)
        newObs.append(dis_goal)
        newObs.append(obsAngle)
        newObs.append(obsRelAngle)
        newObs.append(relAngleDif)
        newObs.append(arrived)

        self.relAngleDif = relAngleDif
        self.goal_dis = dis_goal
        return np.array(newObs)

 
    def draw(self):
        mainColor = [255,0,0]
        lineColor = [255,255,255]

        compassx = self.x + (self.size * math.cos(self.theta))
        compassy = self.y + (self.size * math.sin(self.theta))
        pygame.draw.circle(self.windowSurface, mainColor, (self.x, self.y), self.size)
        pygame.draw.line(self.windowSurface, lineColor, (self.x, self.y), (compassx,compassy))