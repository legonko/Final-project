import RPi.GPIO as GPIO
import time
from threading import Thread, Event

class WheelCounter:
    def __init__(self, but_pin):
        self.but_pin = but_pin
        self.steps = 0
        self.stop_event = Event()
        self.vel = 0

    def counter(self):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.but_pin, GPIO.IN)

        try:
            while not self.stop_event.is_set():
                GPIO.wait_for_edge(self.but_pin, GPIO.FALLING)
                self.steps += 1
        finally:
            GPIO.cleanup()

    def speedometer(self):
        dt = 1
        R = 0.01
        last_x = 0
        
        while not self.stop_event.is_set():
            time.sleep(dt)
            x = self.steps
            diff = x - last_x
            speed = diff / self.dt
            last_x = x
            self.vel = 2 * 3.14 * R * speed
            # print(self.vel, end='\r')

    def start(self):
        self.count_thr = Thread(target=self.counter)
        self.speed_thr = Thread(target=self.speedometer)

        self.count_thr.start()
        self.speed_thr.start()

    def stop(self):
        self.stop_event.set()
        self.count_thr.join()
        self.speed_thr.join()

if __name__ == "__main__":
    but_pin = 22  # Board pin 18
    wheel_counter = WheelCounter(but_pin)
    wheel_counter.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        wheel_counter.stop()
