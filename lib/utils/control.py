import numpy as np
# from jetracer.nvidia_racecar import NvidiaRacecar


def create_car():
    car = NvidiaRacecar()
    car.steering = 0.0
    car.throttle = 0.0
    return car

def steering(steer): #, car
    if steer == 'left':
        # car.steering = -0.3
        print(steer)
    elif steer == 'right':
        # car.steering = 0.3
        print(steer)
    else:
        # car.steering = 0.0
        print(steer)

def move():
    ...