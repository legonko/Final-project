import cv2
import numpy as np
import time
import math
import copy
import lib.utils.config as config
from scipy.signal import find_peaks
from lib.utils import graph_class
from lib.utils.utils import fast_convolution


def find_homography():
    # for single lane
    # for realsense

    x1, y1 = 202, 480
    x2, y2 = 265, 360
    x3, y3 = 360, 360
    x4, y4 = 470, 480

    u1, v1 = x1, y1
    u2, v2 = 202, 0
    u3, v3 = 470, 0
    u4, v4 = x4, y4

    # for cv camera
    # x1, y1 = 210, 480
    # x2, y2 = 300, 315
    # x3, y3 = 400, 315
    # x4, y4 = 590, 480

    # u1, v1 = x1, y1
    # u2, v2 = 210, 0
    # u3, v3 = 500, 0
    # u4, v4 = x4, y4


    # for all road
    '''x1, y1 = 0, 480
    x2, y2 = 150, 360
    x3, y3 = 480, 360
    x4, y4 = 640, 480

    u1, v1 = x1, y1
    u2, v2 = 0, 0
    u3, v3 = 640, 0
    u4, v4 = x4, y4'''
    
    # camera points
    points_camera = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)
    # birds-eye view points
    points_birdseye = np.array([[u1+200, v1+200], [u2+200, v2+200], [u3+200, v3+200], [u4+200, v4+200]], dtype=np.float32)

    homography_matrix, _ = cv2.findHomography(points_camera, points_birdseye)
    
    return homography_matrix


def ipm_ll(image, homography_matrix):
    image = np.asanyarray(image, dtype=np.uint8)
    image = cv2.resize(image, dsize=(640, 480))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    transformed_image = cv2.warpPerspective(image, homography_matrix, (640+400, 480+200)) # cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    return transformed_image

def ipm_pts(pts, homography_matrix):
    transformed_pts = cv2.perspectiveTransform(pts, homography_matrix)
    return transformed_pts

def lanes2map(transformed_image):
    # transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
    # image must be in grascale

    histogram = np.sum(transformed_image, axis=0)
 
    peaks, _ = find_peaks(histogram,  height=4000)  # prominence=10000, prominence - min height above surrounding
    lanes_map = np.zeros_like(transformed_image)
    # lanes_map = cv2.cvtColor(lanes_map, cv2.COLOR_RGB2GRAY)
    peaks = np.sort(peaks)

    if len(peaks) >= 2: # else ??????????????????????????????
        for peak in peaks:
            lanes_map[:, peak] = 255
        # lanes_map[:, peaks[0]] = 100 # left
        # lanes_map[:, peaks[1]] = 250 # right
    print(peaks)
    
    return lanes_map, peaks


def lane_centering(peaks, pos):
    # if pos = (y, x)
    # to do:
    # pos == const == (0, w//2)
    # position == camera position, pos[1] = 640//2
    if len(peaks) == 2: # add supporting for multiple lanes
        rl = abs(pos[1] - peaks[0])
        rr = abs(peaks[1] - pos[1])
        lane_center = int(rl + (rr - rl) / 2)
        # "-" left, "+" right offset from center
        '''
        while loop for lane centering must be in the main function
        '''
        e = pos[0] - lane_center
        if e >= 10:
            steer = 'left'
        elif e < -10:
            steer = 'right'
        else:
            steer = 'straight'
        
    else:
        steer = 'straight'
    
    return steer


def vehicles2map(bounding_boxes, lanes_map):
    '''
    _______________
    |             |  
    |x0,y0   x1,y1|
    ---------------
    '''
    # ipm_map from lanes2map()
    vehicles_map = np.asanyarray(lanes_map)

    for bbox in bounding_boxes:
        c0 = int(bbox[0])
        r0 = int(bbox[1])
        c1 = int(bbox[2])
        r1 = int(bbox[3])
        # r0 = r1
        # print(bbox)
        # l      w         h 
        # 4644 x 1778 x 1482
        # k = l/w = 2.612
        k = 1.2 #2.612
        w = c1 - c0
        y = int(r0 - k * w) #if int(r0 - k * w) >= 0 else 0

        vehicles_map[y:r0, c0:c1] = 255
        print(r0, c0, r1, c1)
        # original bbox
        # c11, c22 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])) 
        # vehicles_map = cv2.cvtColor(vehicles_map, cv2.COLOR_GRAY2BGR)
        # cv2.rectangle(vehicles_map, c11, c22,color=(0,0,255), lineType=cv2.LINE_AA)
    
    return vehicles_map


def scaling(depth_img, bbox_center):
    # m
    horizontal_line_center = (1040 // 2, bbox_center[1]) # (x, y)
    h = 0.15 # height of camera 
    d = get_distance(depth_img, config.depth_scale, bbox_center) # to center of bbox
    L = get_distance(depth_img, config.depth_scale, horizontal_line_center) # to center of horizontal line from bottom bbox line
    l = math.sqrt(d ** 2 - h ** 2) # from ground to center of bbox
    c = math.sqrt(L ** 2 - h ** 2) # from ground to center line
    alpha = math.acos(c / l) # between bbox and vertial line

    return l, alpha

def get_distance(depth_img, point):
    # m
    distance = depth_img[point[1]-15:point[1], point[0]-20:point[0]+20] # x, y
    distance = distance * config.depth_scale
    distance,_,_,_ = cv2.mean(distance)
    
    return distance


def calculate_real_distance(depth_img, bboxes_centers):
    vectors = [] # may be dict: {bbox_center: [l, alpha]...}
    for bbox_center in bboxes_centers:
        l, alpha =  scaling(depth_img, config.depth_scale, bbox_center)
        vectors.append([l, alpha])

    return vectors


def tracking(new_bboxes, old_bboxes):
    # [[x,y]] - center of w coords
    # {(x,y): [(x,y), cost]}
    if old_bboxes is not None:
        new_bboxes[:, 0] = abs(new_bboxes[:, 2] - new_bboxes[:, 0]) # [[xc,yc]]
        new_bboxes = new_bboxes[:, :2]
        old_bboxes[:, 0] = abs(old_bboxes[:, 2] - old_bboxes[:, 0])
        old_bboxes = old_bboxes[:, :2]
        graph = graph_class.Graph()

        for vert1 in new_bboxes:
            for vert2 in old_bboxes:
                graph.add_edge(tuple(vert1), tuple(vert2))

        vertices = list(graph.keys())
        new_graph = graph_class.Graph()
        
        for i in range(len(vertices)):
            neighbours = graph[vertices[i]] 
            min_cost, ind  = min(((item[1], index) for index, item in enumerate(neighbours)), key=lambda x: x[0])
            new_graph.add_edge(vertices[i], neighbours[ind][0])

        # dl = cost
    return new_graph


def calculate_velocity(dt, graph):
    # calculate velocity for each vertex in graph
    vertices = list(graph.keys())
    vel_graph = copy.copy(graph)
    for vert in vertices:
        dl = vel_graph[vert][0][1]
        vel_graph[vert] += [dl / dt] # m/s

    return vel_graph


def create_map(raw_lanes, pos, bboxes, kernel, old_bboxes):
    homography = find_homography()
    # raw_lanes - cv2.cvtColor(raw_lanes, cv2.COLOR_)
    ipm_map = ipm_ll(raw_lanes, homography)
    # cv2.imwrite('ipm_lanes.jpg', ipm_map)
    lanes_map, peaks = lanes2map(ipm_map)
    # cv2.imshow('lanes_map', lanes_map)
    steer = lane_centering(peaks, pos)
    bird_eye_map = vehicles2map(bboxes, lanes_map)
    obstacle_map = vehicles2map(bboxes, np.zeros_like(lanes_map))
    extended_map = fast_convolution(obstacle_map, kernel)

    return bird_eye_map, steer, extended_map