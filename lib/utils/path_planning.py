import math
import time
import numpy as np
from skimage.draw import line
import lib.utils.config as config 
from lib.utils.config_space import create_config_space
from lib.utils.util import get_dist


def path_planer(v, yd=0.4, Ld=1):
    x = np.arange(0, Ld, 0.1)
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


def check_obstacle_static(obstacle_map, angles, v, dt):
    '''angles must be in rad'''
    angles = np.deg2rad(angles)
    current_pos = [480 + config.row_add, (640 + config.column_add) // 2]
    l = v * dt * config.k_pm # vector length
    path = []
    for i in range(len(angles)):
        obstacle_map_expanded = create_config_space(obstacle_map, angles[i])
        next_pos = [int(current_pos[0] - l * math.cos(angles[i])), int(current_pos[1] + l * math.sin(angles[i]))]
        rr, cc = line(*current_pos, *next_pos)
        current_pos = next_pos
        if obstacle_map_expanded[rr, cc] == 255:
            return False

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

def maneuver(v, yd=0.4, Ld=1):
    '''implementation of lane changing'''
    yd = 0.4
    Ld = 1
    X, Y, t = path_planer(v, yd, Ld)
    angles = get_path_angles(X, Y)
    dt = t / len(X)

    return angles, dt