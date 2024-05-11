import os
import cv2
import torch
import io
import json
import zmq
import copy
import time
import pyrealsense2 as rs
import argparse
import onnxruntime as ort
import numpy as np
from lib.core.general import non_max_suppression


def send_array(
        socket, A: np.ndarray, flags: int = 0, copy: bool = True, track: bool = False
    ):
    """send a numpy array with metadata"""
    md = dict(
        dtype=str(A.dtype),
        shape=A.shape,
    )
    socket.send_json(md, flags | zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)


def recv_array(
        socket, flags: int = 0, copy: bool = True, track: bool = False
    ):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    A = np.frombuffer(msg, dtype=md['dtype'])  # type: ignore
    return A.reshape(md['shape'])


def recv_array_and_img(
        socket, flags: int = 0, copy: bool = True, track: bool = False
    ):
    """recv a numpy array"""
    buff = socket.recv()
    img = cv2.imdecode(np.frombuffer(buff, np.uint8), -1)
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    A = np.frombuffer(msg, dtype=md['dtype'])  # type: ignore
    return img, A.reshape(md['shape'])


def deserialize_img(buff):
    return cv2.imdecode(np.frombuffer(buff, np.uint8), -1)


def deserialize_arr(buff):
    memfile = io.BytesIO()
    # If you're deserializing from a bytestring:
    memfile.write(buff)
    # Or if you're deserializing from JSON:
    # memfile.write(json.loads(buff).encode('latin-1'))
    memfile.seek(0)
    return np.load(memfile)


def resize_unscale(img, new_shape=(320, 320), color=114):
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


def infer_yolop():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://192.168.88.42:5555")        
    
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

        color_image = cv2.resize(color_image, (320,320))
        height, width, _ = color_image.shape
        # img_rgb = img_bgr[:, :, ::-1].copy()
        
        # socket.send(b"")
        '''send color_img to server'''
        flag, buff = cv2.imencode(".jpg", color_image)
        socket.send(buff)
        msg1, msg2 = socket.recv_multipart()
        ll_seg_mask = deserialize_img(msg1)
        boxes = deserialize_arr(msg2)
        boxes = copy.copy(boxes)
        ll_seg_mask = copy.copy(ll_seg_mask)
        # cv2.imshow('ll', ll_seg_mask)

        # if len(boxes):
        #     print(f"detect {boxes.shape[0]} bounding boxes.")

        #     img_det = color_image[:, :, ::-1].copy()
            # for *xyxy, conf, _ in boxes:
            #     # print('conf', float(conf.numpy()), type(float(conf.numpy())))
            #     if float(conf) >= 0.60:
            #         x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
            #         img_det = cv2.rectangle(img_det, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
            #         # plot_one_box(xyxy, img_det , line_thickness=2)
            #     else: 
            #         continue
            # for i in range(boxes.shape[0]):
            #     x1, y1, x2, y2, conf, label = boxes[i]
            #     x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
            #     img_det = cv2.rectangle(img_det, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
        
        # ll_seg_out = ll_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
        # ll_seg_mask = np.argmax(ll_seg_out, axis=1)[0]  # (?,?) (0|1)
        # ll_seg_mask = ll_seg_mask * 255
        # ll_seg_mask = ll_seg_mask.astype(np.uint8)
        # ll_seg_mask = cv2.resize(ll_seg_mask, (width, height),
        #                      interpolation=cv2.INTER_LINEAR)
        
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

    infer_yolop()
    """
    PYTHONPATH=. python3 ./test_onnx.py --weight yolop-640-640.onnx --img test.jpg
    """
