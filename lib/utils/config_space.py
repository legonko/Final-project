import numpy as np 
import scipy
from config import w_jr, l_jr, step, angle_step
from lib.utils.util import fast_convolution


def _create_robot_rotates(angle_step=10):
    robot_shape = np.array((int(l_jr/step), int(w_jr/step)))
    robot = np.ones(robot_shape)
    robot = np.pad(robot, int(abs(l_jr-w_jr)/step)+5)
    robot_rotates = [scipy.ndimage.rotate(robot, angle, reshape=False, order=0) for angle in range(0, 180, angle_step)]
    return robot_rotates


def create_config_space(map):
    robot_rotates = _create_robot_rotates(angle_step=angle_step)
    configuration_space = np.zeros((len(robot_rotates),) + map.shape)

    for i, r in enumerate(robot_rotates):
        rot_map = fast_convolution(map, r)
        rot_map[rot_map>0] = 255
        configuration_space[i] = np.copy(rot_map)
        configuration_space = configuration_space.astype(np.uint8)
    
    return configuration_space