import math
import time
import numpy as np
import cv2
import copy
from skimage.draw import line
import lib.utils.config as config 
from lib.utils.config_space import create_config_space
from lib.utils.util import get_dist, velocity_to_control, angle_to_control


def create_path(v, yd=0.4, Ld=1):
    """create path for lane change maneuver"""
    x = np.arange(0, Ld, 0.3)
    Y = yd / (2 * math.pi) * (2 * math.pi * x / Ld - np.sin(2 * math.pi * x / Ld))
    td = Ld / v

    return x, Y, td


def get_point_angle(x, y):
    '''calculate angle b/w vector (x,y) and horizontal line'''
    angle_rad = np.arctan2(y, x)
    angle_deg = np.degrees(angle_rad)
    # if angle_deg < 0:
    #     angle_deg += 360
    return angle_deg


def get_path_angles(x, y):
    '''calculate angles along path'''
    angles = []
    for i in range(1, len(x)):
        angles.append(get_point_angle(x[i] - x[i-1], y[i] - y[i-1]))
    return angles


def check_obstacle_static(obstacle_map, angles, v, dt=0.15):
    '''angles must be in rad'''
    angles = np.deg2rad(angles)
    # beta = np.arctan(self.LB * np.tan(steering) / (self.LB + self.LF))
    current_pos = [config.w + config.row_add - 1, (config.l + config.column_add) // 2 - 1]
    l = v * dt * config.k_pm * 1.6 # vector length
    print('l pp', l)
    obstacle_map_expanded = create_config_space(obstacle_map, 0)
    path = copy.copy(obstacle_map)
    for i in range(len(angles)):
        # obstacle_map_expanded = create_config_space(obstacle_map, angles[i])
        # next pos can be calculated as xc,yc in test_path_planning.ipynb
        next_pos = [int(current_pos[0] - l * math.cos(angles[i])), int(current_pos[1] + l * math.sin(angles[i]))]
        rr, cc = line(*current_pos, *next_pos)
        current_pos = next_pos
        path[rr, cc] = 255
        if np.any(obstacle_map_expanded[rr, cc] == 255):
            return False
        
        #cv2.imwrite('path9.jpg', path)

    return True


def check_obstacle_dynamic(obstacle_map, angles, v, dt):
    '''angles must be in rad'''
    '''angles is dynamic, function must be called in while loop in main'''
    angles = np.deg2rad(angles)
    current_pos = [480 + config.row_add, (640 + config.column_add) // 2] # change to (rows//2, cols//2 on merged frame)
    l = v * dt * config.k_pm # vector length
    path = []
    for i in range(len(angles)):
        obstacle_map_expanded = create_config_space(obstacle_map, angles[i])
        next_pos = [int(current_pos[0] - l * math.cos(angles[i])), int(current_pos[1] + l * math.sin(angles[i]))]
        rr, cc = line(*current_pos, *next_pos)
        if obstacle_map_expanded[rr, cc] == 255:
            return False

    return True


def check_obstacle_xy(obstacle_map, angles, x, y):
    angles = np.deg2rad(angles)
    current_pos = [obstacle_map.shape[0] - 1, obstacle_map.shape[1] // 2 - 1]
    y = current_pos[1] + y * config.k_pm
    x = current_pos[0] - x * config.k_pm
    for i in range(len(angles)):
        obstacle_map_expanded = create_config_space(obstacle_map, angles[i])
        l = get_dist([current_pos[0], current_pos[1]], [x[i], y[i]])
        next_pos = [int(current_pos[0] - l * math.cos(angles[i])), int(current_pos[1] + l * math.sin(angles[i]))]
        rr, cc = line(*current_pos, *next_pos)
        current_pos = next_pos
        if obstacle_map_expanded[rr, cc] == 255:
            return False

    return True


def path_planer(v=1, yd=0.25, Ld=4):
    """create path and calculate heading angles along all path"""
    X, Y, t = create_path(v, yd, Ld)
    print('ttttt', t/len(X))
    angles = get_path_angles(X, Y)
    return angles # np.concatenate((angles, -angles))


def maneuver1(car, angles, v=1):
    """lane change maneuver implementation"""
    # car.throttle = velocity_to_control(v)
    dt = 0.15
    prev_phi = 0
    angles = np.deg2rad(angles)
    for i in range(len(angles)):
        beta = np.arctan(config.LB * np.tan(angles[i]) / (config.LB + config.LF))
        phi = prev_phi + v * dt * np.cos(beta) * np.tan(angles[i]) / (config.LB + config.LF)
        steer_control = angle_to_control(np.rad2deg(phi))
        car.steering = steer_control
        time.sleep(dt)
    # car.steering = 0
    # time.sleep(dt)
    for i in range(len(angles)):
        beta = np.arctan(config.LB * np.tan(-angles[i]) / (config.LB + config.LF))
        phi = prev_phi + v * dt * np.cos(beta) * np.tan(-angles[i]) / (config.LB + config.LF)
        steer_control = angle_to_control(np.rad2deg(phi))
        car.steering = steer_control
        time.sleep(dt)
    
    # car.throttle = 0.0


def maneuver2(car, angles, v):
    # v = 0.185
    L = 0.17 # wheel base
    w = 0.64 * np.tan(np.deg2rad(angles)) / L
    # w = v * np.tan(np.deg2rad(angles)) / L
    dt = 0.1
    angles = np.rad2deg(w*dt)
    print('angles: ', angles)
    for i in range(len(angles)):
        steer_control = angle_to_control(angles[i])
        car.steering = steer_control
        time.sleep(0.1)
    for i in range(len(angles)):
        steer_control = angle_to_control(angles[i])
        car.steering = -steer_control-0.182
        time.sleep(0.1)