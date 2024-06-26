import json
import pyrealsense2 as rs
import numpy as np
import zmq
from lib.utils.util import *
from lib.utils.speedometer import WheelCounter


def control_loop(ctx):
    """Start server"""
    sock = ctx.socket(zmq.REP)
    sock.bind("tcp://0.0.0.0:5678")

    while True:
        cmd = sock.recv_json()
        if cmd['cmd'] == 'exec':
            sock.send(b"")
            exec(cmd['arg'], globals())
        elif cmd['cmd'] == 'eval':
            value = eval(cmd['arg'], globals())
            try:
                s = json.dumps(value)
            except json.JSONDecodeError:
                s = repr(value)
            sock.send_string(s)
        elif cmd['cmd'] == 'color':
            sock.send(dump_jpg(cam.get_image_and_depth()[0]))
        elif cmd['cmd'] == 'depth':
            sock.send(dump_png(cam.get_image_and_depth()[1]))
        elif cmd['cmd'] == 'color_and_depth':
            color_image, depth_image = cam.get_image_and_depth()
            sock.send_multipart([dump_jpg(color_image), dump_png(depth_image)])
        else:
            sock.send(b'')


class RealSense:
    """RealSense D435 class"""
    def __init__(self):
        self.pipe = rs.pipeline()
        self.conf  = rs.config()

        self.conf.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
        self.conf.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)

        self.profile = self.pipe.start(self.conf)
        self.align = rs.align(rs.stream.color)
    
    def get_image_and_depth(self):
        # print(self.profile.get_device())
        frames = self.pipe.wait_for_frames()

        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image


if __name__ == '__main__':
    ctx = zmq.Context()
    cam = RealSense()
    wc = WheelCounter(but_pin='SPI2_MISO')
    wc.start()
    control_loop(ctx)