import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import copy
import numpy as np
from lib.core.general import non_max_suppression
from lib.utils import plot_one_box, show_seg_result
from lib.utils.mapping import *
  

def preprocess_image(img):
    '''
    Preprocess image for YOLOP.onnx

    Args:
        img (np.array): original image
    
    Returns:
        preprocessed image
    '''
    img = cv2.resize(img, (320,320))
    img = img.copy().astype(np.float32)  # (3,640,640) RGB
    img /= 255.0
    img[:, :, 0] -= 0.485
    img[:, :, 1] -= 0.456
    img[:, :, 2] -= 0.406
    img[:, :, 0] /= 0.229
    img[:, :, 1] /= 0.224
    img[:, :, 2] /= 0.225
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)  # (1, 3,640,640)
    return img


def detect(img, ort_session):
    '''
    YOLOP inference

    Args:
        img (np.array): original image 
        ort_session: ort session

    Returns:
        det_out: object detection result
        ll_seg_out: lane detection result
    '''
    preprocessed_img = preprocess_image(img)
    det_out, _, ll_seg_out = ort_session.run(
        ['det_out', 'drive_area_seg', 'lane_line_seg'],
        input_feed={"images": preprocessed_img}
        )
    return det_out, ll_seg_out


def postprocess(color_img, det_out, ll_seg_out):
        '''
        processing results of detection from yolop-320-320.onnx and transform results to 640x480 format

        Args:
            color_img (np.array): original image w/ size (640, 480)
            det_out (arr): bounding boxes, confidence, labels [xyxy,conf,l]
            ll_seg_out (np.array): lane detection result

        Returns:
            img_det (arr): original image w/ bounding boxes and detected lanes w/ size (640,480)
            new_points_arr (np.array): new bounding boxes w/ ipm transformation for size (640,480)
            ll_seg_mask (np.array): lane segmentation mask w/ size (640,480)   
        '''
        det_out = torch.from_numpy(det_out).float()
        boxes = non_max_suppression(det_out, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)[0]  # [n,6] [x1,y1,x2,y2,conf,cls]
        boxes = boxes.cpu().numpy().astype(np.float32)
        
        height, width = 320, 320
        pad_w, pad_h = 0, 0

        ll_predict = ll_seg_out[:, :, pad_h:(height-pad_h), pad_w:(width-pad_w)]
        ll_seg_mask = np.argmax(ll_predict, axis=1)[0]  # (?,?) (0|1)
        ll_seg_mask = ll_seg_mask * 255
        ll_seg_mask = ll_seg_mask.astype(np.uint8)
        ll_seg_mask = cv2.resize(ll_seg_mask, (width, height),
                             interpolation=cv2.INTER_LINEAR)

        ll_seg_mask = cv2.resize(ll_seg_mask, dsize=(640,480))
        _ = None
        img_det = show_seg_result(color_img, (_, ll_seg_mask))
        img_det = cv2.resize(img_det, dsize=(640, 480))

        new_points_arr = []
        new_points = []

        if len(boxes):
            for *xyxy, conf, _ in boxes:
                if float(conf) >= 0.60:
                    xyxy[0] *= 2
                    xyxy[2] *= 2
                    xyxy[1] *= 1.5
                    xyxy[3] *= 1.5
                    plot_one_box(xyxy, img_det , line_thickness=2)
                    xyxy = np.asanyarray(xyxy)
                    points = copy.copy(xyxy)
                    points[1] = points[3]
                    points = points.reshape(-1, 2).reshape(-1, 1, 2)
                    new_points = ipm_pts(points, np.array(config.H))
                    new_points = new_points.reshape(-1, 4)
                    new_points = np.array(new_points)
                    new_points_arr.append(new_points)
                else: 
                    continue

            new_points_arr = np.vstack(new_points_arr) if len(new_points_arr) else None
        else:
            new_points_arr = None

       
        return img_det, new_points_arr, ll_seg_mask  