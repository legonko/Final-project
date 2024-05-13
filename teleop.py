import zmq


jetson_addr = 'tcp://192.168.88.43:5678'



class Car:
    def __init__(self, jetsock: zmq.Socket):
        self.sock = jetsock
        self.exec('car = create_car()')
        self.eval('2+2')
    
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
    def image(self):
        self.eval(f'car')

    @property
    def depth(self):
        self.eval(f'car')


if __name__ == '__main__':
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(jetson_addr)
    car = Car(sock)


