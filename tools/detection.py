import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import copy
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from lib.core.general import non_max_suppression
from lib.utils import plot_one_box, show_seg_result
from lib.utils.mapping import *


def preprocess_image(image):
        '''
        preprocess image for YOLOP pytorch

        Args:
            image (np.array): original image
        
        Returns:
            preprocessed image
        '''
        #image = Image.open(image_path).convert("RGB")
        image = Image.fromarray(image)
        normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
         )
        preprocess = transforms.Compose([
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
            normalize
        ])
        
        return preprocess(image).unsqueeze(0)


def detect(img, model):
    '''
    YOLOP inference

    Args:
        img (np.array): original image 
        model: YOLOP model

    Returns:
        det_out: object detection result
        da_seg_out: drivable area result
        ll_seg_out: lane detection result
    '''
    img = preprocess_image(img)
    # img = preprocess_image(img).to('cuda:0')
    # img = img.half()
    det_out, da_seg_out, ll_seg_out = model(img)
    # print('det_out', det_out)
      
    return det_out, da_seg_out, ll_seg_out
    

def postprocess(color_img, det_out, ll_seg_out):
        '''
        processing results of detection for original YOLOP

        Args:
            color_img (np.array): original image
            det_out: object detection result
            ll_seg_out: lane detection result

        Returns:
            img_det (arr): original image w/ bounding boxes and detected lanes
            new_points_arr (np.array): new bounding boxes w/ ipm transformation
            ll_seg_mask (arr): lane segmentation mask
        '''
        inf_out, _ = det_out
        det_pred = non_max_suppression(inf_out, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)
        det = det_pred[0]

        height, width, _ = color_img.shape
        pad_w, pad_h = 0, 0
        ratio = 1

        ll_predict = ll_seg_out[:, :, pad_h:(height-pad_h), pad_w:(width-pad_w)]

        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

        ll_seg_mask = np.array(ll_seg_mask*255, dtype=np.uint8)
        img_det = show_seg_result(color_img, (_, ll_seg_mask), _, _, is_demo=True)
        img_det = cv2.resize(img_det, dsize=(640, 480))
        # ll_seg_mask = cv2.resize(ll_seg_mask, dsize=(640, 480))

        new_points_arr = []
        new_points = []
        H = find_homography()

        if len(det):
            # print(det)
            for *xyxy, conf, _ in reversed(det):
                # print('conf', float(conf.numpy()), type(float(conf.numpy())))
                if float(conf.cpu().numpy()) >= 0.60:
                    plot_one_box(xyxy, img_det , line_thickness=2)

                    # xyxy = np.asanyarray(xyxy)
                    xyxy = [tensor.cpu().numpy() for tensor in xyxy]
                    xyxy = np.array(xyxy)
                    points = copy.copy(xyxy)
                    # points = np.array(points)
                    # print('points: ', points)
                    points[1] = points[3]
                    points = points.reshape(-1, 2).reshape(-1, 1, 2)
                    # points = points.reshape(-1, 1, 2)
                    new_points = ipm_pts(points, H)
                    # print('_new', new_points)
                    new_points = new_points.reshape(-1, 4)
                    new_points = np.array(new_points)
                    new_points_arr.append(new_points)
                    # print('new', new_points)
                else: 
                    continue

            new_points_arr = np.vstack(new_points_arr) if len(new_points_arr) else None
        else:
            new_points_arr = None

       
        return img_det, new_points_arr, ll_seg_mask    

def postprocess2(color_img, det, ll_seg_mask):
        '''
        processing results of detection from yolop-320-320.onnx and transform results to 640x480 format

        Args:
            color_img (np.array): original image w/ size (640, 480)
            det (arr): bounding boxes, confidence, labels [xyxy,conf,l]
            ll_seg_mask (np.array): lane segmentation mask w/ size (320,320)

        Returns:
            img_det (arr): original image w/ bounding boxes and detected lanes w/ size (640,480)
            new_points_arr (np.array): new bounding boxes w/ ipm transformation for size (640,480)
            ll_seg_mask (np.array): lane segmentation mask w/ size (640,480)   
        '''
        ll_seg_mask = cv2.resize(ll_seg_mask, dsize=(640,480))
        _ = None
        img_det = show_seg_result(color_img, (_, ll_seg_mask), _, _, is_demo=True)
        img_det = cv2.resize(img_det, dsize=(640, 480))

        new_points_arr = []
        new_points = []
        H = config.H

        if len(det):
            for *xyxy, conf, _ in det:
                if float(conf) >= 0.60:
                    xyxy[0] *= 2
                    xyxy[2] *= 2
                    xyxy[1] *= 1.5
                    xyxy[3] *= 1.5
                    plot_one_box(xyxy, img_det , line_thickness=2)
                    # xyxy = [tensor.cpu().numpy() for tensor in xyxy]
                    xyxy = np.asanyarray(xyxy)
                    points = copy.copy(xyxy)
                    # print('points: ', points)
                    points[1] = points[3]
                    points = points.reshape(-1, 2).reshape(-1, 1, 2)
                    # points = points.reshape(-1, 1, 2)
                    new_points = ipm_pts(points, H)
                    # print('_new', new_points)
                    new_points = new_points.reshape(-1, 4)
                    new_points = np.array(new_points)
                    new_points_arr.append(new_points)
                    # print('new', new_points)
                else: 
                    continue

            new_points_arr = np.vstack(new_points_arr) if len(new_points_arr) else None
        else:
            new_points_arr = None

       
        return img_det, new_points_arr, ll_seg_mask    