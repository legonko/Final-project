import math
import time
import numpy as np
from skimage.draw import line
import lib.utils.config as config 
from lib.utils.config_space import create_config_space
from lib.utils.util import angle_to_control


def create_path(v, yd=0.4, Ld=1):
    """create path for lane change maneuver"""
    x = np.arange(0, Ld+0.3, 0.3)
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
    '''calculate heading angles along path'''
    angles = []
    for i in range(1, len(x)):
        angles.append(get_point_angle(x[i] - x[i-1], y[i] - y[i-1]))
    return angles


def check_obstacle_static(obstacle_map, angles, v, dt=0.15):
    """Checks if obstacle is on the path"""
    angles = np.deg2rad(angles)
    current_pos = [config.w + config.row_add - 1, (config.l + config.column_add) // 2 - 1]
    l = v * dt * config.k_pm  # vector length
    obstacle_map_expanded = create_config_space(obstacle_map, 0)
    for i in range(len(angles)):
        obstacle_map_expanded = create_config_space(obstacle_map, -angles[i])
        next_pos = [int(current_pos[0] - l * math.cos(angles[i])), int(current_pos[1] + l * math.sin(angles[i]))]
        rr, cc = line(*current_pos, *next_pos)
        current_pos = next_pos
        if next_pos[0] < 0:
            return True
        if np.any(obstacle_map_expanded[rr[1:], cc[1:]] == 255):
            return False

    return True


def maneuver(car, steerings, dt=0.4):
    """lane change maneuver implementation"""
    steerings = np.rad2deg(steerings)
    print('steerings', steerings)
    for i in range(len(steerings)):
        steer_control = angle_to_control(steerings[i])
        car.steering = steer_control-0.152
        time.sleep(dt)
    car.steering = -0.152


def path_planer(v, yd=0.3, Ld=2):
    """create path and calculate heading and steering angles along all path"""
    x, y, t = create_path(v, yd, Ld)
    print('t: ', t)
    phi = get_path_angles(x, y) # heading
    phi.insert(0, 0.0)
    phi.append(0.0)
    phi2 = []
    for i in range(1, len(phi)):
        phi2.append(phi[i] - phi[i-1])
    ld = 0.3 # lookahead distance
    steerings = np.arctan(2 * config.L * np.sin(np.deg2rad(phi2)) / ld)
    dt = t / (len(x) - 1)

    return phi, steerings, dt