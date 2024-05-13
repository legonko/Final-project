import os
import zmq
import cv2
import torch
import io
import json
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


def send_array_and_img(
        img, socket, A: np.ndarray, flags: int = 0, copy: bool = True, track: bool = False
    ):
    """send a numpy array with metadata"""
    md = dict(
        dtype=str(A.dtype),
        shape=A.shape,
    )
    _, buff = cv2.imencode(".jpg", img)
    socket.send(buff)
    socket.send_json(md, flags | zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)


def dump_img(img):
    _, buff = cv2.imencode(".jpg", img)
    return buff


def deserialize_img(buff):
    return cv2.imdecode(np.frombuffer(buff, np.uint8), -1)


def dump_array(arr):
    memfile = io.BytesIO()
    np.save(memfile, arr)
    serialized = memfile.getvalue()
    # serialized_as_json = json.dumps(serialized.decode('latin-1'))
    return serialized


def recv_array(
        socket, flags: int = 0, copy: bool = True, track: bool = False
    ):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    A = np.frombuffer(msg, dtype=md['dtype'])  # type: ignore
    return A.reshape(md['shape'])


def infer_yolop(weight="yolop-320-320.onnx"):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    ort.set_default_logger_severity(4)
    onnx_path = f"./weights/{weight}"
    ort_session = ort.InferenceSession(onnx_path)

    # device = 'cuda'
    # ort_session.set_providers([f'cuda:{device}'])
    
    # print(f"Load {onnx_path} done!")

    outputs_info = ort_session.get_outputs()
    inputs_info = ort_session.get_inputs()

    for ii in inputs_info:
        print("Input: ", ii)
    for oo in outputs_info:
        print("Output: ", oo)

    print("num outputs: ", len(outputs_info))

    while True:
        '''receive color_img'''
        buff = socket.recv()
        # canvas = cv2.imdecode(np.frombuffer(buff, np.uint8), -1)
        canvas = deserialize_img(buff)
        img_det = canvas[:, :, ::-1].copy()
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
        
        for *xyxy, conf, _ in boxes:
            if float(conf) >= 0.60:
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                img_det = cv2.rectangle(img_det, c1, c2, (255,200,0), thickness=2, lineType=cv2.LINE_AA)
            else: 
                continue
        # cv2.imshow('det', img_det)
        socket.send_multipart([dump_img(ll_seg_mask), dump_array(boxes)])
        # if cv2.waitKey(1) == ord('q'):
        #     break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default="yolop-320-320.onnx")
    parser.add_argument('--img', type=str, default="rs_color_img2.jpg")
    args = parser.parse_args()

    infer_yolop(weight=args.weight)
    """
    PYTHONPATH=. python3 ./test_onnx.py --weight yolop-640-640.onnx --img test.jpg
    """
