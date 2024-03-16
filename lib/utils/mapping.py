import cv2
import numpy as np
from scipy.signal import find_peaks


def ipm_ll(image):
    # for single lane

    x1, y1 = 202, 480
    x2, y2 = 265, 360
    x3, y3 = 360, 360
    x4, y4 = 470, 480

    u1, v1 = x1, y1
    u2, v2 = 202, 0
    u3, v3 = 470, 0
    u4, v4 = x4, y4


    # for all road
    '''x1, y1 = 0, 480
    x2, y2 = 150, 360
    x3, y3 = 480, 360
    x4, y4 = 640, 480

    u1, v1 = x1, y1
    u2, v2 = 0, 0
    u3, v3 = 640, 0
    u4, v4 = x4, y4'''
    
    image = np.asanyarray(image, dtype=np.uint8)
    image = cv2.resize(image, dsize=(640, 480))
    
    # camera points
    points_camera = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)

    # birds-eye view points
    points_birdseye = np.array([[u1, v1], [u2, v2], [u3, v3], [u4, v4]], dtype=np.float32)

    homography_matrix, _ = cv2.findHomography(points_camera, points_birdseye)

    transformed_image = cv2.warpPerspective(image, homography_matrix, (640, 480)) # cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    return transformed_image

def lanes2map(transformed_image):
    # transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
    # image must be in grascale

    histogram = np.sum(transformed_image, axis=0)
 
    peaks, _ = find_peaks(histogram, prominence=10000)  # prominence - min height above surrounding
    lanes_map = np.zeros_like(transformed_image)
    # lanes_map = cv2.cvtColor(lanes_map, cv2.COLOR_RGB2GRAY)
    peaks = np.sort(peaks)

    if len(peaks) >= 2:
        lanes_map[:, peaks[0]] = 100 # left
        lanes_map[:, peaks[1]] = 250 # right
    
    return lanes_map, peaks


def lane_centering(mask, peaks, pos):
    # if pos = (x, y)
    # to do:
    # position == camera position, pos[1] = 640//2
    rl = abs(pos[1] - peaks[0])
    rr = abs(peaks[1] - pos[1])
    lane_center = int(rl + (rr - rl) / 2)
    # "-" left, "+" right offset from center
    e = pos[0] - lane_center
    if e >= 0:
        steering = 'left'
    else:
        steering = 'right'
    
    return steering


def vehicles2map(bounding_boxes, lanes_map):
    '''
    _______________
    |             |  
    |x0,y0   x1,y1|
    ---------------
    '''
    # ipm_map from lanes2map()
    # for ONE bbox

    x0 = bounding_boxes[0]
    y0 = bounding_boxes[3]
    x1 = bounding_boxes[2]
    y1 = bounding_boxes[3]
    # l      w         h 
    # 4644 x 1778 x 1482
    # k = l/w = 2.612
    k = 1.2 #2.612
    w = x1 - x0
    c1 = (int(x0), int(y0 - k * w))
    c2 = (int(x1), int(y1))
    # cv2.rectangle(lanes_map, c1, c2, lineType=cv2.LINE_AA)

    

    vehicles_map = lanes_map
    vehicles_map[c1[1]:y1, c1[0]:x1] = 255
    c11, c22 = (int(bounding_boxes[0]), int(bounding_boxes[1])), (int(bounding_boxes[2]), int(bounding_boxes[3]))
    vehicles_map = cv2.cvtColor(vehicles_map, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vehicles_map, c11, c22,color=(0,0,255), lineType=cv2.LINE_AA)
    
    return vehicles_map


def get_distance(depth_img):
    ...


if __name__ == "__main__":
    img = cv2.imread('rs_color_img2.jpg') # , cv2.IMREAD_GRAYSCALE
    raw_lanes = cv2.imread('test_ll.jpg', cv2.IMREAD_GRAYSCALE)
    ipm = ipm_ll(raw_lanes)
    cv2.imshow('ipm', ipm)
    lanes_map, _ = lanes2map(ipm)
    vehicles_map = vehicles2map([264,329,356,384], lanes_map)

    cv2.imshow('map', vehicles_map) 
    cv2.imshow('origin', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

