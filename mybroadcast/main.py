import logging
import sys
import threading
import time
from typing import TypedDict, Union

import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam


class Resolution(TypedDict):
    width: int
    height: int
    fps: int


class ThreadWithLoggedException(threading.Thread):
    """
    Similar to Thread but will log exceptions to passed logger.

    Args:
        logger: Logger instance used to log any exception in child thread

    Exception is also reachable via <thread>.exception from the main thread.
    """

    def __init__(self, *args, **kwargs):
        try:
            global logger
            self.logger = logger
        except KeyError:
            raise Exception("Missing 'logger' in kwargs")
        super().__init__(*args, **kwargs)
        self.exception = None

    def run(self):
        try:
            if self._target is not None:
                self._target(*self._args, **self._kwargs)
        except Exception as exception:
            thread = threading.current_thread()
            self.exception = exception
            self.logger.exception(
                f"Exception in child thread {thread}: {exception}"
            )
        finally:
            del self._target, self._args, self._kwargs


class CaptureBufferCleanerThread(ThreadWithLoggedException):
    def __init__(self, capture, fps=30, name="capture-buffer-cleaner-thread2"):
        self.capture = capture
        self.last_frame = None
        self.delay = 1 / fps
        super(CaptureBufferCleanerThread, self).__init__(name=name)
        self.success = False
        self.active = True
        self.start()
        self.get_frame = self.get_frame_resize

    def get_next_frame(self):
        self.success, frame = self.capture.read()
        if not self.success:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return self.get_next_frame()
        self.last_frame = frame

    def get_frame_bypass(self):
        return self.last_frame

    def get_frame_resize(self):
        return cv2.resize(self.last_frame, (640, 480))

    def run(self):
        while self.active:
            self.get_next_frame()
            time.sleep(self.delay)


class CromaKeyBackground:
    def __init__(
        self,
        output_settings: Resolution = {"width": 640, "height": 480, "fps": 60},
    ):
        self.frame = np.zeros(
            (output_settings["height"], output_settings["width"], 3), np.uint8
        )  # RGB
        slice_end = slice(0, output_settings["width"])
        self.frame[:, slice_end] = (0, 255, 0)  # B, G, R
        self.active = True

    def get_frame(self):
        return self.frame


# Thread replace background
class ReplaceBackgroundThread(ThreadWithLoggedException):
    def __init__(
        self,
        input,
        background,
        name="replace-background-thread",
        output_settings: Resolution = {"width": 640, "height": 480, "fps": 60},
    ):
        self.background = background
        self.output_settings = output_settings
        self.input = input
        super(ReplaceBackgroundThread, self).__init__(name=name)
        back_frame = np.zeros(
            (output_settings["height"], output_settings["width"], 3), np.uint8
        )  # RGB
        slice_end = slice(0, output_settings["width"])
        back_frame[:, slice_end] = (
            255,
            255,
            255,
        )  # B, G, R
        self.back_frame = back_frame
        self.last_frame = back_frame
        self.mask = back_frame
        self.active = True
        self.selfie_segmentation = (
            mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=1
            )
        )
        self.start()

    def process_frame(self):
        if self.input.success:
            frame = self.input.last_frame
            self.segmentation_results = self.selfie_segmentation.process(frame)
            mask = self.segmentation_results.segmentation_mask
            # mask = cv2.blur(mask, (3, 3))
            frame = frame.astype(np.uint16)
            self.mask = mask
            mask = mask * 16
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = mask.astype(np.uint16)
            pessoa = frame * mask / 16
            mask_ind = mask < 10
            pessoa[mask_ind] = self.background.get_frame()[mask_ind]
            self.last_frame = pessoa.astype(np.uint8)

    def run(self):
        delay = 1 / self.output_settings["fps"]
        while self.active:
            self.process_frame()
            time.sleep(delay)

    def get_frame_rgb(self):
        return cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)


class MyBroadcast:
    background_cleaner: Union[CromaKeyBackground, CaptureBufferCleanerThread]

    def __init__(
        self,
        output_settings: Resolution = {"width": 640, "height": 480, "fps": 30},
        video_path: str = "",
    ) -> None:
        self.output_settings = output_settings
        self.camera = cv2.VideoCapture(1)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera_cleaner = CaptureBufferCleanerThread(self.camera)
        if video_path:
            self.background_cap = cv2.VideoCapture(video_path)
            self.background_cleaner = CaptureBufferCleanerThread(
                self.background_cap
            )
        else:
            self.background_cleaner = CromaKeyBackground()
        self.output1 = ReplaceBackgroundThread(
            input=self.camera_cleaner, background=self.background_cleaner
        )

    def run(self):
        with pyvirtualcam.Camera(width=640, height=480, fps=20) as cam:
            delay = int(1000 / self.output_settings["fps"])
            while True:
                if self.camera_cleaner.last_frame is not None:
                    cv2.imshow("Cam clear", self.camera_cleaner.last_frame)
                    cv2.imshow("Mask", self.output1.mask)
                    cv2.imshow(
                        "Video Background", self.background_cleaner.get_frame()
                    )
                    cv2.imshow("Replace Background", self.output1.last_frame)
                    cam.send(self.output1.get_frame_rgb())
                    cam.sleep_until_next_frame()
                if cv2.waitKey(delay) & 0xFF == ord("q"):
                    break
            self.output1.active = False
            self.background_cleaner.active = False
            self.camera_cleaner.active = False


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    ARGUMENTS_LIST = sys.argv[1:]
    if ARGUMENTS_LIST:
        mb = MyBroadcast(video_path=ARGUMENTS_LIST[0])
    else:
        mb = MyBroadcast()
    mb.run()
