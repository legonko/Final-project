import zmq
import time
import cv2
import onnxruntime as ort
from tools.detection import postprocess2, detect
from lib.utils.util import *
from lib.utils.mapping import *
from lib.utils.control import *
from lib.utils.path_planning import *
from lib.utils import config as config


jetson_addr = 'tcp://192.168.43.2' #'tcp://192.168.88.43'
carport = '5678'
camport = '8765'


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

def main(weight, save_res=False):
    ctx = zmq.Context()
    carsock = ctx.socket(zmq.REQ)
    carsock.connect(jetson_addr + ':' + carport)
    car = Car(carsock)

    ort.set_default_logger_severity(4)
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
    move(car, 0.21)
    '''
    check wheel counter, decrease dt
    measure yd, Ld
    '''
    t_start = time.time()

    if save_res:
        output = cv2.VideoWriter( 
            "exp6.avi", cv2.VideoWriter_fourcc(*'MPEG'), 20, (640, 480)) 
    
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
                car.steering = config.k_steer

        # if time.time() - t_start > 2: #and time.time() - t_start < 3.5
        #     lane_change_flag = True

        # lane change
        if lane_change_flag:
            lane_centering_flag = False
            v = car.speed
            print('v: ', v)
            headings, steerings = path_planer(v, yd=0.3, Ld=2)

            if check_obstacle_static(expanded_map, headings, v, dt=0.35):
                print('lc start')
                maneuver3(car, steerings, dt=0.35)
                print('lc end')
            else:
                print('lc is not possible')
                lane_change_flag = False

            lane_centering_flag = True

        if time.time() - t_start > 9:
            print('end')
            brake(car)
            break

        if save_res:
            output.write(det_img)

        if cv2.waitKey(1) == ord('q'):
            brake(car)
            break

        end_time = time.time()
        old_bboxes = new_bboxes
        print('fps: ', 1/ (end_time-start_time))

    cv2.destroyAllWindows() 
    if save_res: 
        output.release() 
    if KeyboardInterrupt:
        brake(car)


if __name__ == '__main__':
    main(weight="yolop-320-320.onnx", save_res=True)