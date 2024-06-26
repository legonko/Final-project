import cv2
import numpy as np


def show_seg_result(img, result):
    """Draw lanes segmentation"""
    color_area = np.zeros((result[1].shape[0], result[1].shape[1], 3), dtype=np.uint8)
    color_area[result[1] == 255] = [0, 255, 0]
    color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    color_mask = np.mean(color_seg, 2)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    img = img.astype(np.uint8)
 
    return img

    
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """Plots one bounding box on image"""
    tl = line_thickness or round(0.0001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or (255,200,0)
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    '''
    _______________
    |x0,x1        |  
    |        x2,x3|
    ---------------
    '''
    # Start coordinate represents the top left corner of rectangle 
    # Ending coordinate represents the bottom right corner of rectangle
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)