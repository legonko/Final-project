import os
import cv2
import torch
import time
import pyrealsense2 as rs
import argparse
import onnxruntime as ort
import numpy as np
from lib.core.general import non_max_suppression


def resize_unscale(img, new_shape=(640, 640), color=114):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    canvas = np.zeros((new_shape[0], new_shape[1], 3))
    canvas.fill(color)
    # Scale ratio (new / old) new_shape(h,w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w,h
    new_unpad_w = new_unpad[0]
    new_unpad_h = new_unpad[1]
    pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h  # wh padding

    dw = pad_w // 2  # divide padding into 2 sides
    dh = pad_h // 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img

    return canvas, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)


def infer_yolop(weight="yolop-320-320.onnx",
                img_path="rs_color_img2.jpg"):

    ort.set_default_logger_severity(4)
    onnx_path = f"./weights/{weight}"
    ort_session = ort.InferenceSession(onnx_path)
    print(f"Load {onnx_path} done!")

    outputs_info = ort_session.get_outputs()
    inputs_info = ort_session.get_inputs()

    for ii in inputs_info:
        print("Input: ", ii)
    for oo in outputs_info:
        print("Output: ", oo)

    print("num outputs: ", len(outputs_info))
    
    pipe = rs.pipeline()
    cnfg  = rs.config()

    cnfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
    cnfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)

    profile = pipe.start(cnfg)
    # align depth to rgb
    align = rs.align(rs.stream.color)


    while True:
        start_time = time.time()
        frames = pipe.wait_for_frames()

        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not color_frame or not depth_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        

        depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,
                                        alpha = 0.5), cv2.COLORMAP_JET)
        cv2.resize(color_image, (320,320))
        height, width, _ = color_image.shape
        # img_rgb = img_bgr[:, :, ::-1].copy()
        canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(color_image, (320, 320))

        img = canvas.copy().astype(np.float32)  # (3,640,640) RGB
        img /= 255.0
        img[:, :, 0] -= 0.485
        img[:, :, 1] -= 0.456
        img[:, :, 2] -= 0.406
        img[:, :, 0] /= 0.229
        img[:, :, 1] /= 0.224
        img[:, :, 2] /= 0.225

        img = img.transpose(2, 0, 1)

        img = np.expand_dims(img, 0)  # (1, 3,640,640)

        det_out, _, ll_seg_out = ort_session.run(
        ['det_out', 'drive_area_seg', 'lane_line_seg'],
        input_feed={"images": img}
        )

        det_out = torch.from_numpy(det_out).float()
        boxes = non_max_suppression(det_out)[0]  # [n,6] [x1,y1,x2,y2,conf,cls]
        boxes = boxes.cpu().numpy().astype(np.float32)

        if boxes.shape[0] == 0:
            print("no bounding boxes detected.")

        # scale coords to original size.
        boxes[:, 0] -= dw
        boxes[:, 1] -= dh
        boxes[:, 2] -= dw
        boxes[:, 3] -= dh
        boxes[:, :4] /= r

        print(f"detect {boxes.shape[0]} bounding boxes.")

        img_det = color_image[:, :, ::-1].copy()
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2, conf, label = boxes[i]
            x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
            img_det = cv2.rectangle(img_det, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
        
        ll_seg_out = ll_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
        ll_seg_mask = np.argmax(ll_seg_out, axis=1)[0]  # (?,?) (0|1)
        ll_seg_mask = ll_seg_mask * 255
        ll_seg_mask = ll_seg_mask.astype(np.uint8)
        ll_seg_mask = cv2.resize(ll_seg_mask, (width, height),
                             interpolation=cv2.INTER_LINEAR)
        
        # cv2.imshow('img_det', img_det)
        # cv2.imshow('ll_seg_mask', ll_seg_mask)

        if cv2.waitKey(1) == ord('q'):
            break

        end_time = time.time()
        print('fps: ', 1/ (end_time-start_time))

    pipe.stop()
    cv2.destroyAllWindows() 
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default="yolop-320-320.onnx")
    parser.add_argument('--img', type=str, default="rs_color_img2.jpg")
    args = parser.parse_args()

    infer_yolop(weight=args.weight, img_path=args.img)
    """
    PYTHONPATH=. python3 ./test_onnx.py --weight yolop-640-640.onnx --img test.jpg
    """
