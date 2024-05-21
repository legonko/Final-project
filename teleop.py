import zmq
import time
import cv2
import numpy as np
import onnxruntime as ort
from tools.detection import postprocess2
from lib.utils.util import *
from lib.utils.mapping import *
# from lib.utils.control import *
from lib.utils.path_planning import *


jetson_addr = 'tcp://192.168.43.2' #'tcp://192.168.88.43'
carport = '5678'
camport = '8765'

def preprocess_image(img):
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
    preprocessed_img = preprocess_image(img)
    det_out, _, ll_seg_out = ort_session.run(
        ['det_out', 'drive_area_seg', 'lane_line_seg'],
        input_feed={"images": preprocessed_img}
        )
    return det_out, ll_seg_out
    

def move(car, value):
    # throttle_control = velocity_to_control(value)
    car.steering = -0.192
    car.throttle = value


def brake(car):
    car.throttle = 0.0


def lane_centering_steering(car, d):
    steer_control = angle_to_control(5)
    car.steering = -d*steer_control
    time.sleep(0.1)
    car.steering = 0-0.162
    time.sleep(0.1)
    car.steering = d*steer_control
    time.sleep(0.1)
    car.steering = 0-0.162


class RemoteObject:
    def exec(self, expr):
        self.sock.send_json({
            "cmd": "exec",
            "arg": expr
        })
        self.sock.recv()
    
    def eval(self, expr):
        self.sock.send_json({
            "cmd": "eval",
            "arg": expr
        })
        return self.sock.recv_json()

class DepthCam(RemoteObject):
    def __init__(self, camsock: zmq.Socket):
        self.sock = camsock
    
    def recv(self, id: str):
        self.sock.send_string(id)
        data = self.sock.recv()
        return load_img(data)
    
    @property
    def color(self):
        return self.recv('color')
    
    @property
    def depth(self):
        return self.recv('depth')
        
        

class Car(RemoteObject):
    def __init__(self, jetsock: zmq.Socket):
        self.sock = jetsock
        self.exec('car = create_car()')
    
    def __del__(self):
        self.exec('del car')
    
    def exec(self, expr):
        self.sock.send_json({
            "cmd": "exec",
            "arg": expr
        })
        self.sock.recv()
    
    def eval(self, expr):
        self.sock.send_json({
            "cmd": "eval",
            "arg": expr
        })
        return self.sock.recv_json()
    
    def load_image(self, tok):
        self.sock.send_json({
            'cmd': tok
        })
        return load_img(self.sock.recv())
    
    def load_image_mp(self, tok):
        self.sock.send_json({
            'cmd': tok
        })
        clr, dpt = self.sock.recv_multipart()
        return load_img(clr), load_img(dpt) 

    @property
    def throttle(self):
        return self.eval('car.throttle')

    @throttle.setter
    def throttle(self, value):
        self.exec(f'car.throttle = {value}')
    
    @property
    def steering(self):
        return self.eval('car.steering')
    
    @steering.setter
    def steering(self, value):
        self.exec(f'car.steering = {value}')

    @property
    def speed(self):
        return self.eval('wc.vel')

    @property
    def color(self):
        return self.load_image('color')

    @property
    def depth(self):
        return self.load_image('depth')
    
    @property
    def color_and_depth(self):
        return self.load_image_mp('color_and_depth')

def main():
    ctx = zmq.Context()
    carsock = ctx.socket(zmq.REQ)
    carsock.connect(jetson_addr + ':' + carport)
    car = Car(carsock)

    ort.set_default_logger_severity(4)
    weight = "yolop-320-320.onnx"
    onnx_path = f"./weights/{weight}"
    ort_session = ort.InferenceSession(onnx_path)

    # device = 'cuda'
    # ort_session.set_providers([f'cuda:{device}'])
    
    old_bboxes = None

    t0 = time.time()
    color_image, _ = car.color_and_depth
    
    det_out, ll_seg_out = detect(color_image, ort_session)
    _, old_bboxes, _ = postprocess2(color_image, det_out, ll_seg_out)
    
    lane_change_flag = False
    lane_centering_flag = True
    move(car, 0.2) #0.35
    t_start = time.time()

    output = cv2.VideoWriter( 
        "output5.avi", cv2.VideoWriter_fourcc(*'MPEG'), 20, (640, 480)) 
    
    while True:
        start_time = time.time()
        dt = time.time() - t0
        t0 = time.time()
        # t_start_img = time.time()
        color_image, depth_image = car.color_and_depth
        # print('recv time: ', time.time()-t_start_img)
        # t_start_det = time.time()
        det_out, ll_seg_out = detect(color_image, ort_session)
        # print('detect time: ', time.time() - t_start_det)
        # t_start_post = time.time()
        det_img, new_bboxes, ll_seg_mask = postprocess2(color_image, det_out, ll_seg_out)
        # print('postprocess time: ', time.time() - t_start_post)
        # t_start_map = time.time()
        steer, expanded_map = create_map(ll_seg_mask, new_bboxes, depth_image, dt, old_bboxes)
        # print('map time: ', time.time() - t_start_map)
        # lane centering
        if lane_centering_flag:
            print(steer)
            if steer == 'left':
                lane_centering_steering(car, d=1)
            elif steer == 'right':
                lane_centering_steering(car, d=-1)
            elif steer == 'straight':
                car.steering = 0 - 0.172

        if time.time() - t_start > 2 and time.time() - t_start < 3.5:
            lane_change_flag = True
            # lane_centering_flag = False

        # path planning
        if lane_change_flag:
            v = car.speed
            print('v: ', v)
            headings, steerings = path_planer(v, yd=0.3, Ld=2, dt=0.4)

            if check_obstacle_static(expanded_map, headings, v):
                print('lc start')
                maneuver3(car, steerings, dt=0.4)
                print('lc end')
                time.sleep(2)
                brake(car)
                break
            else:
                print('lc is not possible')
                lane_change_flag = False
                lane_centering_flag = True
            lane_centering_flag = True

        if time.time() - t_start > 9:
            print('end')
            brake(car)
            break

        # cv2.imshow('ipm', cv2.resize(det_ipm, (640, 480)))
        # cv2.imshow('source', det_img)
        output.write(det_img)
        # cv2.imshow('bev', cv2.resize(bird_eye_map, (640, 480)))
        # cv2.imshow('expanded_map', expanded_map)

        if cv2.waitKey(1) == ord('q'):
            brake(car)
            break

        end_time = time.time()
        old_bboxes = new_bboxes
        print('fps: ', 1/ (end_time-start_time))

    cv2.destroyAllWindows() 
    output.release() 
    if KeyboardInterrupt:
        brake(car)


if __name__ == '__main__':
    main()