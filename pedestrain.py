import pygame
import math
from vec2d import vec2d


class Creep():

    updatesPerSecond = 10
    realTimeFactor =  10
    color = (40, 140, 40) 
    def __init__(self,
                 surface,
                 radius,
                 pos_init,
                 dir_init,
                 speed,
                 SCREEN_WIDTH,
                 SCREEN_HEIGHT,
                 jitter_speed):

        self.windowSurface = surface
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.BOXWIDTH = 100
        self.jitter_speed = jitter_speed
        self.speed = speed
        self.radius = radius
        self.pos = vec2d(pos_init)
        self.direction = vec2d(dir_init)
        self.direction.normalize()  # normalized


    def update(self):

        dx = self.direction.x * self.speed 
        dy = self.direction.y * self.speed

        if self.pos.x + dx > self.SCREEN_WIDTH - self.BOXWIDTH - self.radius:
            self.pos.x = self.SCREEN_WIDTH - self.radius - self.BOXWIDTH
            self.direction.x = -1 * self.direction.x *  (1 + 0.5 * self.jitter_speed)  # a little jitter
        elif self.pos.x + dx <= self.radius + self.BOXWIDTH :
            self.pos.x = self.radius + self.BOXWIDTH
            self.direction.x = -1 * self.direction.x * (1 + 0.5 * self.jitter_speed)  # a little jitter
        else:
            self.pos.x = self.pos.x + dx

        if self.pos.y + dy > self.SCREEN_HEIGHT  - self.BOXWIDTH - self.radius:
            self.pos.y = self.SCREEN_HEIGHT - self.radius - self.BOXWIDTH
            self.direction.y = -1 * self.direction.y * (1 + 0.5 * self.jitter_speed)  # a little jitter
        elif self.pos.y + dy <= self.radius + self.BOXWIDTH :
            self.pos.y = self.radius + self.BOXWIDTH
            self.direction.y = -1 * self.direction.y * (1 + 0.5 * self.jitter_speed)  # a little jitter
        else:
            self.pos.y = self.pos.y + dy

        self.direction.normalize()


    def draw(self):

        lineColor = [255,255,255]
        
        pygame.draw.circle(self.windowSurface,self.color, (int(self.pos.x),int(self.pos.y)), self.radius)
        theta = math.atan2(self.direction.y,self.direction.x)
        compassx = self.pos.x + (self.radius * math.cos(theta))
        compassy = self.pos.y + (self.radius * math.sin(theta))
        pygame.draw.line(self.windowSurface, lineColor, (self.pos.x, self.pos.y), (compassx,compassy))