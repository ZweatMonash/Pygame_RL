import math
import pygame
import random

def drawGoal(surface,goal):
    goalColor = [0, 50 , 0]
    for i in range(3):
        #goalColor[0] = goalColor[0] + 50*i
        goalColor[1] = goalColor[1] + 50*i
        #goalColor[2] = goalColor[2] - 50*i
        pygame.draw.circle(surface, goalColor, goal['position'], goal['size']/(i+1))

def drawObstacles(surface, agentColor, obsList):
    for shape in obsList:
        pygame.draw.polygon(surface, agentColor, shape)

def checkCollision(laserScanData,minScanRange):
    for scan in laserScanData:
        if scan <= float(minScanRange)/100:
            return True
    return False


def isColor(RGB1,RGB2):
    if(RGB1[0] == RGB2[0] and RGB1[1] == RGB2[1] and RGB1[2] == RGB2[2]):
        return True
    else:
        return False

def getInterAgentDistace(position1,position2):
    distance = math.hypot(position1[0] - position2[0], position1[1] - position2[1])
    return distance

def getInterAgentTheta(position1,position2):
    theta = math.atan2(position2[1] - position1[1],position2[0] - position1[0])
    return theta



def centreRecttoPoly(objDat):
    position = objDat['position']
    width = objDat['width']
    height = objDat['height']

    point1 = [position[0] - width/2, position[1] - height/2]
    point2 = [position[0] + width/2, position[1] - height/2]
    point3 = [position[0] + width/2, position[1] + height/2]
    point4 = [position[0] - width/2, position[1] + height/2]

    return [point1,point2,point3,point4]

def cornerRecttoPoly(objDat):
    position = objDat['position']
    width = objDat['width']
    height = objDat['height']

    point1 = [position[0], position[1]]
    point2 = [position[0] + width, position[1]]
    point3 = [position[0] + width, position[1] + height]
    point4 = [position[0], position[1] + height]

    return [point1,point2,point3,point4]

def blackWithinRadius(surface, xpos, ypos, radius):
# any obstacles within a radius - 
    containsBlack = False
    checkRadius = radius
    checkPoints = 10 #20 COVERS better

    if not isColor(surface.get_at((xpos,ypos))[:3],[0,0,0]):
        for i in range(checkPoints):
            Ang = (2*math.pi/float(checkPoints)) * i
            checkX = int(xpos + (float(radius) * math.cos(Ang)))
            checkY = int(ypos+ (float(radius) * math.sin(Ang)))
            if isColor(surface.get_at((checkX,checkY))[:3],[0,0,0]):
                containsBlack = True
                break
        if not containsBlack:
            checkRadius = radius*2/3
            for i in range(checkPoints):
                Ang = (2*math.pi/float(checkPoints)) * i
                checkX = int(xpos + (float(radius) * math.cos(Ang)))
                checkY = int(ypos+ (float(radius) * math.sin(Ang)))
                if isColor(surface.get_at((checkX,checkY))[:3],[0,0,0]):
                    containsBlack = True
                    break
        if not containsBlack:
            checkRadius = radius/3
            for i in range(checkPoints):
                Ang = (2*math.pi/float(checkPoints)) * i
                checkX = int(xpos + (float(radius) * math.cos(Ang)))
                checkY = int(ypos+ (float(radius) * math.sin(Ang)))
                if isColor(surface.get_at((checkX,checkY))[:3],[0,0,0]):
                    containsBlack = True
                    break
    else:
        containsBlack = True

    return containsBlack

def noCollideSpawnCheck(xPos, yPos, blackListOjects, safeDistance):
    safe = True
    for i in range(len(blackListOjects)):
        black_x = blackListOjects[i][0]
        black_y = blackListOjects[i][1]
        if math.hypot(xPos - black_x , yPos - black_y) < safeDistance:
            safe = False
    return safe

def noCollideSpawn(surface, blacklist,distance):
    maxX = surface.get_width() - 200
    maxY = surface.get_height() - 200


    objX = int(random.uniform(200, maxX))
    objY = int(random.uniform(200, maxY))
    tryDist = distance
    attemptCounter = 0
    maxAttempts = 1000
    while not noCollideSpawnCheck(objX,objY,blacklist,tryDist) or blackWithinRadius(surface, objX, objY, 100):
        objX = int(random.uniform(200, maxX))
        objY = int(random.uniform(200, maxY))
        attemptCounter += 1
        if attemptCounter > maxAttempts:
            print(str(tryDist))
            tryDist = (0.75) * float(tryDist)
            print('Error: Tried ' + str(maxAttempts) + ' spawn locations, reducing safe distance to ' + str(tryDist))
            attemptCounter = 0
            if tryDist <= 50:
                print('failed to find good spawn point')
                break

    return objX,objY
