"""
@author: Trinh Khai Truong 
"""
import cv2
import threading

class WebcamStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise IOError("Cannot open webcam")
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            frame = self.frame.copy() if self.ret else None
        return frame

    def stop(self):
        self.stopped = True
        self.stream.release()