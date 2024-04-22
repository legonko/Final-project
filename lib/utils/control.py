import numpy as np
# from jetracer.nvidia_racecar import NvidiaRacecar
from . import util as util


def create_car():
    car = NvidiaRacecar()
    car.steering = 0.0
    car.throttle = 0.0
    return car

def steering(car, angle): #, car
    steer_control = util.angle_to_control(angle)
    car.steering = steer_control

def move(car, v):
    throttle_control = util.velocity_to_control(v)
    car.throttle = throttle_control

def brake(car):
    car.throttle = 0.0