import zmq

from lib.utils.util import load_img


jetson_addr = 'tcp://192.168.88.43'
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
        self.eval('speed')

    @property
    def color(self):
        return self.load_image('color')

    @property
    def depth(self):
        return self.load_image('depth')


if __name__ == '__main__':
    ctx = zmq.Context()
    carsock = ctx.socket(zmq.REQ)
    carsock.connect(jetson_addr + ':' + carport)
    car = Car(carsock)
    car.steering = 0.5 - car.steering
    img = car.color
    import matplotlib.pyplot as plt
    plt.imshow(img); plt.show()


