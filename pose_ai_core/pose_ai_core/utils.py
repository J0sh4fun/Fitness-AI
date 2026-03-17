import cv2
import numpy as np
from threading import Thread, Lock

def midpoint(pt1, pt2):
    """Return the midpoint between two (x, y) points."""
    return np.array([
        (pt1[0] + pt2[0]) / 2,
        (pt1[1] + pt2[1]) / 2
    ])

class VideoReader:
    """Multi-threaded reader for video files or camera streams."""
    def __init__(self, file_name, width=640, height=480):
        try:
            self.file_name = int(file_name)
        except ValueError:
            self.file_name = file_name
        
        self.cap = cv2.VideoCapture(self.file_name)
        # Set resolution (Lowering this significantly boosts FPS)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.grabbed, self.frame = self.cap.read()
        self.read_lock = Lock()
        self.stopped = False

    def __iter__(self):
        # Start the background thread
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        """Continuously grab frames from the camera."""
        while not self.stopped:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            if not grabbed:
                self.stop()

    def __next__(self):
        with self.read_lock:
            if not self.grabbed or self.stopped:
                raise StopIteration
            return self.frame.copy()

    def stop(self):
        self.stopped = True
        self.cap.release()

# BGR color palette
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]