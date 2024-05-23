import copy
import io
import cv2
import zmq
import numpy as np
from scipy.signal import fftconvolve
import re
import math
from . import config


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def get_dist(p1, p2):
    '''get distance b/w two points'''
    return np.linalg.norm(
        np.array(p1) - np.array(p2)
    )

def load_img(buff):
    return cv2.imdecode(np.frombuffer(buff, np.uint8), -1)


def deserialize_arr(buff):
    memfile = io.BytesIO()
    # If you're deserializing from a bytestring:
    memfile.write(buff)
    # Or if you're deserializing from JSON:
    # memfile.write(json.loads(buff).encode('latin-1'))
    memfile.seek(0)
    return np.load(memfile)


def dump_jpg(img):
    _, buff = cv2.imencode(".jpg", img)
    return buff

def dump_png(img):
    _, buff = cv2.imencode('.png', img)
    return buff

def dump_array(arr):
    memfile = io.BytesIO()
    np.save(memfile, arr)
    serialized = memfile.getvalue()
    # serialized_as_json = json.dumps(serialized.decode('latin-1'))
    return serialized


def recv_array(socket, flags: int = 0, copy: bool = True, track: bool = False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    A = np.frombuffer(msg, dtype=md['dtype'])  # type: ignore
    return A.reshape(md['shape'])


def send_array(socket, A: np.ndarray, flags: int = 0, copy: bool = True, track: bool = False):
    """send a numpy array with metadata"""
    md = dict(
        dtype=str(A.dtype),
        shape=A.shape,
    )
    socket.send_json(md, flags | zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)


def angle_to_control(angle_degrees):
    '''convert angle (deg) to control signal ([-1;1])'''
    max_angle_degrees = 25
    # print('angle_degrees', angle_degrees, type(angle_degrees))
    # print('max_angle_degrees', max_angle_degrees, type(max_angle_degrees))
    control_signal = angle_degrees / max_angle_degrees
    return control_signal


def velocity_to_control(velocity):
    '''convert velocity (m/s) to control signal ([-1;1])'''
    max_velocity = 5.56 
    control_signal = velocity / max_velocity
    return control_signal


def merge_frames(frame_front, frame_back):
    '''merge frames from front and back cameras'''
    frame_back = frame_back[::-1] # vertical mirror
    frame_back = frame_back[:, ::-1] # horizontal mirror
    merged_frame = np.concatenate((frame_front, frame_back), axis=0)
    return merged_frame


def point_mirror_vertical(point):
    """mirror point along vertival axis"""
    new_vert = [config.l + config.column_add - point[0] - 1, point[1]]
    return new_vert


def point_mirror_horizontal(point):
    """mirror point along horizontal axis"""
    new_hor = [point[0], config.w + config.row_add - point[1] - 1]
    return new_hor


def point_mirror(point):
    """mirror point along vertival & horizontal axes"""
    point_vert = point_mirror_vertical(point)
    point_hor = point_mirror_horizontal(point_vert)
    return point_hor


def bbox_mirror(bbox):
    """mirror bounding box along vertival & horizontal axes"""
    for i in range(bbox.shape[0]):
        bbox[i, :] = point_mirror(bbox[i, 2:]) + point_mirror(bbox[i, :2])
    return bbox


def recalculate_coords(bbox):
    """recalculate coordinates of bounding box for merged frame"""
    for i in range(bbox.shape[0]):
        bbox[i, 1] = bbox[i, 1] + config.w + config.row_add - 1
        bbox[i, 3] = bbox[i, 3] + config.w + config.row_add - 1
    return bbox

def recalculate_coords_graph(vel_graph):
    '''mirror and transform for merged frame coordinates (x1, y1, x2, y2) in vel_graph'''
    vertices = list(vel_graph.keys())
    for vert in vertices:
        new_vert = tuple(bbox_mirror(np.array(copy.copy(vert))))
        vel_graph[new_vert] = vel_graph.pop(vert)
        new_vert2 = tuple(recalculate_coords(np.array([copy.copy(vert)])))
        vel_graph[new_vert2] = vel_graph.pop(new_vert)
    return vel_graph


def calculate_vector_difference(l1, l2, phi1, phi2):
    '''
    l1 - from ground center to car on first frame (meters)
    l2 - from ground center to car on second frame
    phi - angle between vertical and vector l (rad)
    '''
    x1 = l1 * math.cos(math.pi - phi1)
    y1 = l1 * math.sin(math.pi - phi1)
    x2 = l2 * math.cos(math.pi - phi2)
    y2 = l2 * math.sin(math.pi - phi2)
    # delta = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    delta = abs(y2 - y1)

    return delta


def calculate_vector_difference2(l1, l2, coords1, coords2):
    '''
    calculate k between pixels and meters
    l_pixels, l_meters --> k (meters/pixels)
    find dl in pixels, then convert to meters
    l1, l2 - meters
    coords - coordinates of bboxes
    '''
    x1, y1 = coords1[0], coords1[1]
    x2, y2 = coords2[0], coords2[1]
    xc, yc = (config.l+config.column_add)//2, config.w+config.row_add
    delta_pixels = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    l_pixels_1 = math.sqrt((xc - x1) ** 2 + (yc - y1) ** 2)
    l_pixels_2 = math.sqrt((xc - x2) ** 2 + (yc - y2) ** 2)
    k = (l1 / l_pixels_1 + l2 / l_pixels_2) / 2
    delta = k * delta_pixels
    return delta


def fast_convolution(img, kernel):
    res = fftconvolve(img, kernel, mode='same')
    res[res>255] = 255
    return res.astype(np.uint8)


def offset_calculation(mpu):
    offsets = [[0.0],[0.0],[0.0]]
    buffer_x = []
    buffer_y = []
    buffer_z = []
    for i in range(100):
        # accel_data = mpu.get_gyro_data()
        # buffer_x.append(accel_data['x'])
        # buffer_y.append(accel_data['y'])
        # buffer_z.append(accel_data['z'])
        gyro_data = mpu.get_gyro_data()
                # Угловая скорость
        gx = 1000 * gyro_data['x'] / 32768
        gy = 1000 * gyro_data['y'] / 32768
        gz = 1000 * gyro_data['z'] / 32768
        buffer_x.append(gx)
        buffer_y.append(gy)
        buffer_z.append(gz)
    offsets[0] = sum(buffer_x) / 100
    offsets[1] = sum(buffer_y) / 100
    offsets[2] = sum(buffer_z) / 100

    return offsets