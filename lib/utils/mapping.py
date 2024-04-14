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
    # for realsense

    # x1, y1 = 202, 480
    # x2, y2 = 265, 360
    # x3, y3 = 360, 360
    # x4, y4 = 470, 480

    # u1, v1 = x1, y1
    # u2, v2 = 202, 0
    # u3, v3 = 470, 0
    # u4, v4 = x4, y4

    # for cv camera
    # x1, y1 = 210, 480
    # x2, y2 = 300, 315
    # x3, y3 = 400, 315
    # x4, y4 = 590, 480

    # u1, v1 = x1, y1
    # u2, v2 = 210, 0
    # u3, v3 = 500, 0
    # u4, v4 = x4, y4
    
    # for 13 cm height
    x1, y1 = 290, 480
    x2, y2 = 300, 380
    x3, y3 = 360, 380
    x4, y4 = 370, 480

    u1, v1 = x1, y1
    u2, v2 = 275, 0
    u3, v3 = 370, 0
    u4, v4 = x4, y4
    
    # camera points
    points_camera = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)
    # birds-eye view points
    points_birdseye = np.array([[u1+config.column_add//2, v1+config.row_add], 
                                [u2+config.column_add//2, v2+config.row_add], 
                                [u3+config.column_add//2, v3+config.row_add], 
                                [u4+config.column_add//2, v4+config.row_add]], dtype=np.float32)

    homography_matrix, _ = cv2.findHomography(points_camera, points_birdseye)
    
    return homography_matrix


def find_inverse_homography():
    x1, y1 = 290, 480
    x2, y2 = 300, 380
    x3, y3 = 360, 380
    x4, y4 = 370, 480

    u1, v1 = x1, y1
    u2, v2 = 275, 0
    u3, v3 = 370, 0
    u4, v4 = x4, y4
    
    # camera points
    points_camera = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)
    # birds-eye view points
    points_birdseye = np.array([[u1+config.column_add//2, v1+config.row_add], 
                                [u2+config.column_add//2, v2+config.row_add], 
                                [u3+config.column_add//2, v3+config.row_add], 
                                [u4+config.column_add//2, v4+config.row_add]], dtype=np.float32)

    homography_matrix, _ = cv2.findHomography(points_birdseye, points_camera)
    
    return homography_matrix


def ipm_ll(image, homography_matrix):
    image = np.asanyarray(image, dtype=np.uint8)
    image = cv2.resize(image, dsize=(640, 480))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    transformed_image = cv2.warpPerspective(image, homography_matrix, (640+config.column_add, 480+config.row_add)) # cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    return transformed_image


def ipm_pts(pts, homography_matrix):
    transformed_pts = cv2.perspectiveTransform(pts, homography_matrix)
    return transformed_pts


def lanes2map(transformed_image):
    # image must be in grascale
    histogram = np.sum(transformed_image, axis=0)
 
    peaks, _ = find_peaks(histogram,  prominence=8000)  # prominence=10000, prominence - min height above surrounding
    lanes_map = np.zeros_like(transformed_image)
    peaks = np.sort(peaks)
    print('peaks', peaks)

    if len(peaks): # else ??????????????????????????????
        for peak in peaks:
            lanes_map[:, peak] = 255
        # lanes_map[:, peaks[0]] = 100 # left
        # lanes_map[:, peaks[1]] = 250 # right
    # print('peaks', peaks)
    
    return lanes_map, peaks


def lane_centering(peaks):
    # if pos = (y, x)
    # pos == const == (0, w//2) == camera position
    '''what if there is only 1 peak????????'''
    pos = config.pos
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
    vehicles_map = np.asanyarray(lanes_map)

    for bbox in bounding_boxes:
        c0 = int(bbox[0])
        r0 = int(bbox[1])
        c1 = int(bbox[2])
        r1 = int(bbox[3])
        '''
        r0 = r1
        l       w       h 
        4644 x 1778 x 1482
        '''
        k = 1.2 #2.612
        w = c1 - c0
        y = int(r0 - k * w) #if int(r0 - k * w) >= 0 else 0
        vehicles_map[y:r0, c0:c1] = 255
        # print(r0, c0, r1, c1)
    
    return vehicles_map


def scaling(depth_img, bbox_center):

    '''only if use add_edge4'''
    bbox_center = list(bbox_center)
    bbox_center[0] = bbox_center[0] + abs(bbox_center[2] - bbox_center[0]) // 2
    bbox_center = bbox_center[:2]

    # bbox_center[0] = bbox_center[0]-400
    # bbox_center[1] = bbox_center[1]-200
    horizontal_line_center = np.array([(640+config.column_add) // 2, bbox_center[1]]) # (x, y)
    # print('horizont: ', horizontal_line_center)
    h = 0.13 # height of camera 
    d = get_distance(depth_img, bbox_center) # to center of bbox
    # L = get_distance(depth_img, horizontal_line_center) # to center of horizontal line from bottom bbox line
    # print('bbox', bbox_center)
    # print('d: ', d, 'L: ', L)
    l = math.sqrt(d ** 2 - h ** 2) # from ground to center of bbox
    print('l: ', l)
    # c = math.sqrt(L ** 2 - h ** 2) # from ground to center line
    # print('c: ', c)
    # alpha = math.acos(c / l) if l > c else 0 # between bbox and vertial line

    xc, yc = (640+config.column_add)//2, 480+config.row_add # bottom center
    c = abs(bbox_center[0] - xc) # from bbox center to intersection b/w horizont and vertical
    L = abs(bbox_center[1] - yc) # from bottom center to intersection b/w horizont and vertical
    # c, L in pixels !!!
    alpha = math.atan2(c, L)

    # alpha is in radians !!!
    return l, alpha

def get_distance(depth_img, point):
    # m
    # points = point.reshape(-1, 2)
    points = copy.copy(np.array(point))
    points = points.reshape(1, 1, 2)
    # print('_points', points, 'shape', points.shape)
    points = ipm_pts(points, find_inverse_homography())
    points = points.reshape(-1, 2).flatten()
    points = np.array(points)
    # print('points', points, 'shape', points.shape)
    # distance = depth_img[int(point[1]-10):int(point[1]), int(point[0]-5):int(point[0]+5)] # x, y
    distance = depth_img[int(points[1]), int(points[0])]
    if distance == 0.0:
        # distance = depth_img[int(point[1]-10):int(point[1]), int(point[0]-5):int(point[0]+5)]
        # distance,_,_,_ = cv2.mean(distance)
        distance = depth_img[int(points[1]), int(points[0]-10)]

    distance = distance * config.depth_scale
    # distance,_,_,_ = cv2.mean(distance)
    
    return distance


def tracking(new_bboxes, old_bboxes, depth_img):
    # [[x,y]] - center of w coords
    # {(x,y): [(x,y), cost]}
    # print('__nb: ', new_bboxes)
    # print('__ob: ', old_bboxes)
    if old_bboxes is not None:
        # new_bboxes[:, 0] = new_bboxes[:, 0] + abs(new_bboxes[:, 2] - new_bboxes[:, 0]) // 2 # not true -> [[xc,yc]]
        # new_bboxes = new_bboxes[:, :2]
        # old_bboxes[:, 0] = old_bboxes[:, 0] + abs(old_bboxes[:, 2] - old_bboxes[:, 0]) // 2 # true -> r, c
        # old_bboxes = old_bboxes[:, :2]
        graph = graph_class.Graph()
        print('nb: ', new_bboxes)
        print('ob: ', old_bboxes)

        for vert1 in new_bboxes:
            for vert2 in old_bboxes:
                l1, phi1 = scaling(depth_img, copy.copy(vert1))
                # print('l1: ', l1, 'phi1: ', phi1)
                l2, phi2 = scaling(depth_img, copy.copy(vert2))
                # print('l2: ', l2, 'phi2: ', phi2)
                # graph.add_edge(tuple(vert1), tuple(vert2))
                graph.add_edge2(vert1, vert2, l2, l1, phi2, phi1)
                # graph.add_edge3(vert1, vert2, l1, l2)
                # graph.add_edge4(vert1, vert2, l1, l2) # checked

        vertices = list(graph.keys())
        new_graph = graph_class.Graph()
        
        for i in range(len(vertices)):
            neighbours = graph[vertices[i]] 
            min_cost, ind  = min(((item[1], index) for index, item in enumerate(neighbours)), key=lambda x: x[0])
            # new_graph.add_edge(vertices[i], neighbours[ind][0])
            l1, phi1 = scaling(depth_img, copy.copy(vertices[i]))
            l2, phi2 = scaling(depth_img, copy.copy(neighbours[ind][0]))
            new_graph.add_edge2(vertices[i], neighbours[ind][0], l2, l1, phi2, phi1)
            # new_graph.add_edge3(vertices[i], neighbours[ind][0], l1, l2)
            # new_graph.add_edge4(vertices[i], neighbours[ind][0], l1, l2) # checked
        # print('new graph: ', new_graph)

        return new_graph
    else:
        return None


def calculate_velocity(dt, graph):
    '''calculate velocity for each vertex in graph'''
    vertices = list(graph.keys())
    vel_graph = copy.copy(graph)
    for vert in vertices:
        dl =  vel_graph[vert][0][1] # add if not use depth: config.step *
        vel_graph[vert] += [dl / dt] # m/s

    return vel_graph


def test_func(nbb, obb, dt, depth_img):
    ng = tracking(nbb, obb, depth_img)
    if ng is not None:
        vel_gr = calculate_velocity(dt, ng) # checked
        print('vel graph', vel_gr)
        return vel_gr
    else:
        return None


def predict_trajectory():
    '''predict cars position with current_pos and dl
    work with graph'''
    ...

def expand(vel_graph, vehicle_map, t=20):
    '''expand map with velocity and t needed for lane change'''
    vertices = list(vel_graph.keys())
    expanded_map = np.zeros_like(vehicle_map)
    k = 1.2
    for vert in vertices:
        print('vert', vert)
        w = abs(vert[0] - vert[2])
        y = int(vert[1] - k * w)
        # expanded_map[y:vert[1], vert[0]:vert[2]] = 255
        dl =  vel_graph[vert][1] * t # add [0] at the end if don't works
        print('dl', dl)
        expanded_map[int(y-dl):int(vert[1]), int(vert[0]):int(vert[2])] = 255

    return expanded_map
    
def create_map(raw_lanes, bboxes, kernel, det_img, depth_img=None, dt=None, old_bboxes=None):
    ''''
    old: vel graph {(667.7803, 245.60457): [((782.91364, 272.1528), 0.006153875828001754), 0.005138973870901224]}
    '''
    H = find_homography()
    ipm_map = ipm_ll(raw_lanes, H)
    det_ipm = ipm_ll(det_img, H)
    lanes_map, peaks = lanes2map(ipm_map)
    steer = lane_centering(peaks)

    if bboxes is not None:
        bird_eye_map = vehicles2map(bboxes, lanes_map)
        obstacle_map = vehicles2map(bboxes, np.zeros_like(lanes_map))
        vel_graph = test_func(bboxes, old_bboxes, dt, depth_img)
        if vel_graph is not None:
            expanded_map2 = expand(vel_graph, obstacle_map)
        else:
            expanded_map2 = np.zeros_like(lanes_map)
        # print('source boxes', bboxes, old_bboxes)
        # depth_img = cv2.medianBlur(depth_img, 17)
        '''try this for blur: dtype=np.float32'''
        '''check rotations in config space'''
        '''expand map along y, make map scaling'''
        
    else:
        bird_eye_map = lanes_map
        obstacle_map = np.zeros_like(lanes_map)
        expanded_map2 = np.zeros_like(lanes_map)

    expanded_map = fast_convolution(obstacle_map, kernel)

    return bird_eye_map, steer, expanded_map2, lanes_map, det_ipm