import numpy as np 
import scipy
from .config import w_jr, l_jr, step, angle_step
from lib.utils.util import fast_convolution


def _create_robot_rotates(angle=0):
    robot_shape = np.array((int(l_jr/step), int(w_jr/step)))
    robot = np.ones(robot_shape)
    robot = np.pad(robot, int(abs(l_jr-w_jr)/step)+5)
    robot_rotate = scipy.ndimage.rotate(robot, angle, reshape=False, order=0)
    return robot_rotate


def create_config_space(map, angle=0):
    robot_rotate = _create_robot_rotates(angle)
    configuration_space = fast_convolution(map, robot_rotate)
    configuration_space[configuration_space>0] = 255
    configuration_space = configuration_space.astype(np.uint8)
    
    return configuration_space