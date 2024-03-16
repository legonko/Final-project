import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from lib.core.general import non_max_suppression
from lib.utils import plot_one_box, show_seg_result
from lib.utils.plot import plot_subbox
from lib.utils.mapping import ipm_ll
from lib.core.postprocess import morphological_process,  connect_lane


def preprocess_image(image):
        #image = Image.open(image_path).convert("RGB")
        image = Image.fromarray(image)
        preprocess = transforms.Compose([
            transforms.Resize((480, 640)),
            transforms.ToTensor()
        ])
        return preprocess(image).unsqueeze(0)


def detect(img, model):
    img = preprocess_image(img)
    det_out, da_seg_out, ll_seg_out = model(img)
      
    return det_out, da_seg_out, ll_seg_out
    

def process(color_img,  det_out, da_seg_out, ll_seg_out): # second pos was: depth_img,
        '''
        conf thres
        det_out[0:2]
        '''
        #1
        # inf_out = det_out[0]
        # print(det_out[1]) # [0] - 5,  
        # det_pred = non_max_suppression(inf_out, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)
        
        # det = det_pred[0]
        inf_out, _ = det_out
        det_pred = non_max_suppression(inf_out, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)
        det=det_pred[0]

        height, width, _ = color_img.shape
        pad_w, pad_h = 0, 0
        ratio = 1

        da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)

        '''
        pad_h =  0
        pad_w =  0
        ratio =  1.0
        '''

        ll_predict = ll_seg_out[:, :, pad_h:(height-pad_h), pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

        # use raw data without shitty postprocess
        
        # ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
        # ll_seg_mask = connect_lane(ll_seg_mask)
        # ipm_img = ipm_ll(ll_seg_mask*255)
       # cv2.imwrite('ipm_ll_seg.jpg', ipm_img)
       # ll_map = lanes2map(ipm_img)


        img_det = show_seg_result(color_img, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)
        img_det = cv2.resize(img_det, dsize=(640, 480))
        # depth_img = cv2.resize(depth_img, dsize=(640, 480))

        if len(det):
           # det[:,:4] = scale_coords(color_img.shape[2:],det[:,:4],img_det.shape).round()
            for *xyxy,conf,cls in reversed(det):
                plot_one_box(xyxy, img_det , line_thickness=2)
                print(xyxy)
               # plot_subbox(xyxy, img_det)

                # plot_one_box(xyxy, depth_img , line_thickness=3, color=(0,0,0))
                # plot_subbox(xyxy, depth_img)
              
        
       
        return img_det, ll_seg_mask #ipm_img, depth_img
      #  cv2.imshow('image', img_det)
       # cv2.imshow('map', ll_map)
       
