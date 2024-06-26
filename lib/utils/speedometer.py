import RPi.GPIO as GPIO
import time
from threading import Thread, Event

class WheelCounter:
    def __init__(self, but_pin):
        self.but_pin = but_pin
        self.steps = 0
        self.vel = 0
        self.stop_event = Event()

    def counter(self):
        GPIO.setmode(GPIO.TEGRA_SOC)
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
            speed = diff / dt
            last_x = x
            self.vel = 2 * 3.14 * R * speed # linear velocity in m/s

    def start(self):
        self.count_thr = Thread(target=self.counter)
        self.speed_thr = Thread(target=self.speedometer)

        self.count_thr.start()
        self.speed_thr.start()

    def stop(self):
        self.stop_event.set()
        self.count_thr.join()
        self.speed_thr.join()
