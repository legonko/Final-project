import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import copy
import numpy as np
import torchvision.transforms as transforms
import lib.utils.config as config
from PIL import Image

from lib.core.general import non_max_suppression
from lib.utils import plot_one_box, show_seg_result
from lib.utils.mapping import *


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
    

def postprocess(color_img,  det_out, ll_seg_out):#old_bboxes
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
        det = det_pred[0]

        height, width, _ = color_img.shape
        pad_w, pad_h = 0, 0
        ratio = 1

        # da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        # da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        # _, da_seg_mask = torch.max(da_seg_mask, 1)
        # da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)

        '''
        pad_h =  0
        pad_w =  0
        ratio =  1.0
        '''

        ll_predict = ll_seg_out[:, :, pad_h:(height-pad_h), pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

        # if len(det):
        #     xyxy = reversed(det)[:, :4]
        #     xyxy = np.asanyarray(xyxy)
        #     points = copy.copy(xyxy)
        #     # print('points: ', points)
        #     points[:, 1] = points[:, 3]
        #     points = points.reshape(-1, 2)
        #     points = points.reshape(-1,1,2)
        #     new_points = ipm_pts(points, find_homography())
        #     # if new_points is not None:
        #     new_points = new_points.reshape(-1, 4)
        #     new_points = np.array(new_points)
        #     # print('new points: ', new_points)
        # else:
        #      new_points = None


        ll_seg_mask = np.array(ll_seg_mask*255, dtype=np.uint8)
        # kernel = np.ones((int(config.l_jr // config.step), int(config.w_jr // config.step)))
        # bird_eye_map, steer, extended_map = create_map(ll_seg_mask, config.pos, new_points, kernel) # old_bboxes


        img_det = show_seg_result(color_img, (_, ll_seg_mask), _, _, is_demo=True)
        img_det = cv2.resize(img_det, dsize=(640, 480))

        new_points_arr = []

        if len(det):
            
            for *xyxy, conf, _ in reversed(det):
                # print('conf', float(conf.numpy()), type(float(conf.numpy())))
                if float(conf.numpy()) >= 0.6:
                    plot_one_box(xyxy, img_det , line_thickness=2)

                    xyxy = np.asanyarray(xyxy)
                    print(xyxy)
                    points = copy.copy(xyxy)
                    # print('points: ', points)
                    points[1] = points[3]
                    points = points.reshape(-1, 2)
                    points = points.reshape(-1, 1, 2)
                    new_points = ipm_pts(points, find_homography())
                    print('_new', new_points)
                    new_points = new_points.reshape(-1, 4)
                    new_points = np.array(new_points)
                    new_points_arr.append(new_points)
                    print('new', new_points)
                else: 
                    continue

            new_points_arr = np.vstack(new_points_arr)
        else:
            new_points_arr = None

        
        
       
        return img_det, new_points_arr, ll_seg_mask    