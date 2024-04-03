import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import time
from detection import detect, postprocess
import argparse
from lib.config import cfg
from lib.utils.utils import create_logger, select_device
from lib.models import get_net
from PIL import Image
from matplotlib.pyplot import imshow
from lib.utils.mapping import *
from lib.utils.control import *


def load_model():
    logger, _, _ = create_logger(
    cfg, cfg.LOG_DIR, 'demo')

    device = select_device(logger,opt.device)

    # Load model
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location= device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    model.eval()

    return model

def rs_stream(model):
    # createing car
    # car = create_car()
    
    pipe = rs.pipeline()
    cnfg  = rs.config()

    cnfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
    cnfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)

    profile = pipe.start(cnfg)
    # align depth to rgb
    align = rs.align(rs.stream.color)
    # depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    old_bboxes = None
    kernel = np.ones((int(config.l_jr // config.step), int(config.w_jr // config.step)))

    frames = pipe.wait_for_frames()
    time.sleep(5)
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    t0 = time.time()
    color_image = np.asanyarray(color_frame.get_data())
    # cv2.imwrite('test_lab.jpg', color_image)
    det_out, _, ll_seg_out = detect(color_image, model)
    _, old_bboxes, _ = postprocess(color_image, det_out, ll_seg_out)


    while True:
        start_time = time.time()
        frames = pipe.wait_for_frames()

        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        dt = time.time() - t0 #if t1 != None else None
        t0 = time.time()

        if not color_frame or not depth_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        

        depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,
                                        alpha = 0.5), cv2.COLORMAP_JET)

        det_out, _, ll_seg_out = detect(color_image, model)
        det_img, new_bboxes, ll_seg_mask = postprocess(color_image, det_out, ll_seg_out)
        bird_eye_map, steer, expanded_map, l_map, det_ipm = create_map(ll_seg_mask, new_bboxes, kernel, det_img, depth_image, dt, old_bboxes)
        # steering(steer, car)
        # add PID control
        # path planning

        #cv2.imshow('rgb', color_image)
        # cv2.imshow('rgb', ipm_img)
        cv2.imshow('source', det_img)
        cv2.imshow('bev', bird_eye_map)
       
        # cv2.imshow('detected', det_ipm)
        cv2.imshow('expanded_map', expanded_map)

        if cv2.waitKey(1) == ord('q'):
            break

        end_time = time.time()
        old_bboxes = new_bboxes

    pipe.stop()
    cv2.destroyAllWindows() 

def put_img(model, frame):
    frame = np.asanyarray(frame)
    frame = cv2.resize(frame, dsize=(640, 480))
    frame = np.asanyarray(frame)
    kernel = np.ones((int(config.l_jr // config.step), int(config.w_jr // config.step)))

    det_out, da_seg_out, ll_seg_out = detect(frame, model)
    # det_img, bird_eye_map, steer, expanded_map = postprocess(frame, det_out, da_seg_out, ll_seg_out)
    det_img, new_bboxes, ll_seg_mask = postprocess(frame, det_out, ll_seg_out, old_bboxes)
    bird_eye_map, steer, expanded_map = create_map(ll_seg_mask, new_bboxes, kernel, old_bboxes)
    #det_img = Image.fromarray(det_img)
    # cv2.imwrite('test_ll.jpg', ll_seg_mask*255)
    # ll = ll_seg_mask*255
    # ll = cv2.cvtColor(ll, cv2.COLOR_GRAY2BGR)
    # det_img_1 = cv2.cvtColor(det_img, cv2.COLOR_RGB2BGR)
    # steering(steer)
    
    while True:
        cv2.imshow('detection', det_img)
        cv2.imshow('bev', bird_eye_map) 
        cv2.imshow('extended', expanded_map) 
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    cv2.destroyAllWindows()
    

def cv_stream(model):

    vid = cv2.VideoCapture(0) 
    kernel = np.ones((int(config.l_jr // config.step), int(config.w_jr // config.step)))
    
    while(True): 
        
        ret, frame = vid.read() 
        frame = cv2.resize(frame, dsize=(640, 480))
        frame = np.asanyarray(frame)
        # cv2.imwrite('cv_frame.jpg', frame)

        det_out, _, ll_seg_out = detect(frame, model)
        # det_img, bird_eye_map, steer = postprocess(frame, det_out, da_seg_out, ll_seg_out)
        det_img, new_bboxes, ll_seg_mask = postprocess(frame, det_out, ll_seg_out)
        bird_eye_map, steer, expanded_map, lanes_map, det_ipm = create_map(ll_seg_mask, new_bboxes, kernel, det_img)
        # steering(steer)
        
        
        cv2.imshow('rgb', det_img) 
        cv2.imshow('bev', bird_eye_map)
        cv2.imshow('det ipm', det_ipm)
        # cv2.imshow('extended', expanded_map)
        

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            # cv2.imwrite('report_source.jpg', frame)
            # cv2.imwrite('report_det_img.jpg', det_img)
            # cv2.imwrite('report_bev.jpg', bird_eye_map)
            # cv2.imwrite('report_expanded.jpg', expanded_map)
            # cv2.imwrite('report_raw_lanes.jpg', ll_seg_mask)
            break
    
    vid.release() 
    cv2.destroyAllWindows() 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/videos', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        model = load_model()
        rs_stream(model)
        # cv_stream(model)
        # img = Image.open('rs_color_img2.jpg').convert("RGB")  # rs_color_img2.jpg
        # put_img(model, img)
        cv2.destroyAllWindows()
