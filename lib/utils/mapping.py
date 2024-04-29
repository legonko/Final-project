import cv2
import numpy as np
import time
import math
import copy
import lib.utils.config as config
from scipy.signal import find_peaks
from lib.utils import graph_class
from lib.utils.util import merge_frames, bbox_mirror, recalculate_coords, recalculate_coords_graph
from lib.utils.util import fast_convolution
from lib.utils.config_space import create_config_space


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
    transformed_image = cv2.warpPerspective(image, homography_matrix, (config.l+config.column_add, config.w+config.row_add))

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
    # print('peaks', peaks)

    if len(peaks):
        for peak in peaks:
            lanes_map[:, peak] = 255
    
    return lanes_map, peaks


def lane_centering(peaks):
    # pos == const == (w//2, 0) == camera position
    pos = config.pos
    if len(peaks):
        delta = pos[0] - peaks
        left_closest = pos[0] - np.min(delta[delta > 0]) if len(delta[delta > 0]) else None
        right_closest =  pos[0] - np.max(delta[delta < 0]) if len(delta[delta < 0]) else None
        if left_closest and right_closest:
            lane_center = int(left_closest + (right_closest - left_closest) / 2)
            e = pos[0] - lane_center

            if e >= 10:
                steer = -12.0
            elif e < -10:
                steer = 12.0
            else:
                steer = 0.0
        else:
            steer = 0.0
        
    else:
        steer = 0.0
    
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
    
    return vehicles_map


def scaling(depth_img, bbox_center):
    '''only if use add_edge4'''
    bbox_center = list(bbox_center)
    bbox_center[0] = bbox_center[0] + abs(bbox_center[2] - bbox_center[0]) // 2
    bbox_center = bbox_center[:2]

    h = 0.13 # height of camera 
    d = get_distance(depth_img, bbox_center) # to center of bbox
    l = math.sqrt(d ** 2 - h ** 2) # from ground to center of bbox
    print('l: ', l)
    xc, yc = (config.l + config.column_add) // 2, config.w + config.row_add # bottom center
    c = abs(bbox_center[0] - xc) # from bbox center to intersection b/w horizont and vertical
    L = abs(bbox_center[1] - yc) # from bottom center to intersection b/w horizont and vertical
    # c, L in pixels !!!
    alpha = math.atan2(c, L)

    # alpha is in radians !!!
    return l, alpha

def get_distance(depth_img, point):
    '''get distance from camera to point im meters'''
    points = copy.copy(np.array(point))
    points = points.reshape(1, 1, 2)
    # print('_points', points, 'shape', points.shape)
    points = ipm_pts(points, find_inverse_homography())
    points = points.reshape(-1, 2).flatten()
    points = np.array(points)
    # print('points', points, 'shape', points.shape)
    distance = depth_img[int(points[1]), int(points[0])]
    if distance == 0.0:
        # distance = depth_img[int(point[1]-10):int(point[1]), int(point[0]-5):int(point[0]+5)]
        # distance,_,_,_ = cv2.mean(distance)
        distance = depth_img[int(points[1]), int(points[0]-10)]

    distance = distance * config.depth_scale
    # distance,_,_,_ = cv2.mean(distance)
    
    return distance


def tracking(new_bboxes, old_bboxes, depth_img):
    '''tracking all obstacles through their centers'''
    # [[x,y]] - center of w coords
    # {(x,y): [(x,y), cost]}
    if old_bboxes is not None:
        graph = graph_class.Graph()
        print('nb: ', new_bboxes)
        print('ob: ', old_bboxes)

        for vert1 in new_bboxes:
            for vert2 in old_bboxes:
                l1, phi1 = scaling(depth_img, copy.copy(vert1))
                # print('l1: ', l1, 'phi1: ', phi1)
                l2, phi2 = scaling(depth_img, copy.copy(vert2))
                # print('l2: ', l2, 'phi2: ', phi2)
                graph.add_edge2(vert1, vert2, l2, l1, phi2, phi1)

        vertices = list(graph.keys())
        new_graph = graph_class.Graph()
        
        for i in range(len(vertices)):
            neighbours = graph[vertices[i]] 
            min_cost, ind  = min(((item[1], index) for index, item in enumerate(neighbours)), key=lambda x: x[0])
            l1, phi1 = scaling(depth_img, copy.copy(vertices[i]))
            l2, phi2 = scaling(depth_img, copy.copy(neighbours[ind][0]))
            new_graph.add_edge2(vertices[i], neighbours[ind][0], l2, l1, phi2, phi1)
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


def expand(vel_graph, t=20): # vehicle_map, 
    '''expand map with velocity and t needed for lane change'''
    vertices = list(vel_graph.keys())
    expanded_map = np.zeros((config.w+config.row_add, config.l+config.column_add)) # np.zeros_like(vehicle_map) 
    k = 1.2
    for vert in vertices:
        print('vert', vert)
        w = abs(vert[0] - vert[2])
        y = int(vert[1] - k * w)
        # expanded_map[y:vert[1], vert[0]:vert[2]] = 255
        dl =  vel_graph[vert][1] * t
        dl *= config.k_pm # m * pixels/m
        print('dl', dl)
        expanded_map[int(y-dl):int(vert[1]), int(vert[0]):int(vert[2])] = 255

    return expanded_map
    

def create_map(raw_lanes, bboxes, kernel, det_img, depth_img=None, dt=None, old_bboxes=None):
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
            expanded_map2 = expand(vel_graph)
        else:
            expanded_map2 = obstacle_map
        # depth_img = cv2.medianBlur(depth_img, 17)
        '''try this for blur: dtype=np.float32'''
        
    else:
        bird_eye_map = lanes_map
        obstacle_map = np.zeros_like(lanes_map)
        expanded_map2 = obstacle_map

    current_angle = 0
    expanded_map = create_config_space(obstacle_map, current_angle)

    return bird_eye_map, steer, expanded_map2, lanes_map, det_ipm


def process_frame(H, raw_lanes, bboxes, old_bboxes, depth_img, dt):
    ipm_map = ipm_ll(raw_lanes, H)
    # det_ipm = ipm_ll(det_img, H)
    lanes_map, peaks = lanes2map(ipm_map)
    steer = lane_centering(peaks)
    if bboxes is not None:
        bird_eye_map = vehicles2map(bboxes, lanes_map)
        obstacle_map = vehicles2map(bboxes, np.zeros_like(lanes_map))
        vel_graph = test_func(bboxes, old_bboxes, dt, depth_img)
        if vel_graph is not None:
            expanded_map_vel = expand(vel_graph, obstacle_map)
        else:
            expanded_map_vel = obstacle_map
        
    else:
        bird_eye_map = lanes_map
        obstacle_map = np.zeros_like(lanes_map)
        expanded_map_vel = obstacle_map

    return bird_eye_map, steer, expanded_map_vel


def create_map2(data, dt, current_angle):
    H = find_homography()
    raw_lanes1, bboxes1, old_bboxes1, depth_img1, raw_lanes2, bboxes2, old_bboxes2, depth_img2 = data
    bird_eye_map1, steer1, expanded_map_vel1 = process_frame(H, 
                                                            raw_lanes1,  
                                                            bboxes1, 
                                                            old_bboxes1, 
                                                            depth_img1, 
                                                            dt
                                                            )
    bird_eye_map2, _, expanded_map_vel2 = process_frame(H, 
                                                        raw_lanes2, 
                                                        bboxes2, 
                                                        old_bboxes2, 
                                                        depth_img2, 
                                                        dt
                                                        )    
    
    merged_map = merge_frames(expanded_map_vel1, expanded_map_vel2)
    expanded_map = create_config_space(merged_map, current_angle)

    return expanded_map, steer1



def process_frame_merged(H, raw_lanes, bboxes, old_bboxes, depth_img, dt):
    ipm_map = ipm_ll(raw_lanes, H)
    lanes_map, peaks = lanes2map(ipm_map)
    steer = lane_centering(peaks)
    if bboxes is not None:
        bird_eye_map = vehicles2map(bboxes, lanes_map)
        obstacle_map = vehicles2map(bboxes, np.zeros_like(lanes_map))
        vel_graph = test_func(bboxes, old_bboxes, dt, depth_img)
    else:
        bird_eye_map = lanes_map
        obstacle_map = np.zeros_like(lanes_map)
        vel_graph = None

    return bird_eye_map, steer, vel_graph, obstacle_map


def create_map_merged(data, dt, current_angle):
    H = find_homography()
    raw_lanes1, bboxes1, old_bboxes1, depth_img1, raw_lanes2, bboxes2, old_bboxes2, depth_img2 = data
    # ipm_map1 = ipm_ll(raw_lanes1, H)
    # ipm_map2 = ipm_ll(raw_lanes2, H)
    # merged_ipm = merge_frames(ipm_map1, ipm_map2)
    # lanes_map, peaks = lanes2map(merged_ipm) # fix lanes_map
    # steer = lane_centering(peaks) # fix

    # bboxes2 = bbox_mirror(bboxes2)
    # old_bboxes2 = bbox_mirror(old_bboxes2)
    # bboxes2 = recalculate_coords(bboxes2)
    # old_bboxes2 = recalculate_coords(old_bboxes2)

    # bboxes = np.vstack((bboxes1, bboxes2)) # bb1 + bb2
    # old_bboxes = np.vstack((old_bboxes1, old_bboxes2))
    bird_eye_map1, steer1, vel_graph1, obstacle_map1 = process_frame_merged(H, 
                                                            raw_lanes1,  
                                                            bboxes1, 
                                                            old_bboxes1, 
                                                            depth_img1, 
                                                            dt
                                                            )
    bird_eye_map2, _, vel_graph2, obstacle_map2 = process_frame_merged(H, 
                                                        raw_lanes2, 
                                                        bboxes2, 
                                                        old_bboxes2, 
                                                        depth_img2, 
                                                        dt
                                                        ) 
    # transform coords in vel_graph2
    vel_graph2 = recalculate_coords_graph(vel_graph2)

    if vel_graph1 is None:
        if vel_graph2 is None:
            vel_graph = None
        else:
            vel_graph = vel_graph2
    else:
        vel_graph = {**vel_graph1, **vel_graph2}

    # merged_map = merge_frames(expanded_map_vel1, expanded_map_vel2)
    
    if vel_graph is not None:
        expanded_map_vel = expand(vel_graph, obstacle_map)
    else:
        expanded_map_vel = obstacle_map


# y in first vel_graph has changed on merged map !!!!!!!! check x, y and r,c
   
    
    
    # expanded_map = create_config_space(merged_map, current_angle)

    return expanded_map, steer1