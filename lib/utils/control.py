import time
from jetracer.nvidia_racecar import NvidiaRacecar
from . import util as util
from . import config as config


def create_car():
    car = NvidiaRacecar()
    car.steering = 0.0 + config.k_steer
    car.throttle = 0.0
    return car

def move(car, value):
    car.steering = config.k_steer
    car.throttle = value

def brake(car):
    car.throttle = 0.0

def lane_centering_steering(car, d):
    steer_control = util.angle_to_control(5)
    car.steering = -d*steer_control
    time.sleep(0.1)
    car.steering = 0 + config.k_steer
    time.sleep(0.1)
    car.steering = d*steer_control
    time.sleep(0.1)
    car.steering = 0 + config.k_steer