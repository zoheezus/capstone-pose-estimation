import logging
import threading
import traceback
from queue import Queue

import cv2
from tf_pose_estimation.tf_pose.estimator import TfPoseEstimator
from tf_pose_estimation.tf_pose.networks import get_graph_path, model_wh

import time
import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PoseAppWSockets():

    def __init__(self, camera=0, resize='0x0', resize_out_ratio=4.0, model="mobilenet_thin", show_process=False,
               remote_server='', delay_time=500):

        self.last_epoch = time.time()
        self.delay_time = delay_time
        self.remote_server = remote_server
        self.show_process = show_process
        self.model = model
        self.resize = resize
        self.resize_out_ratio = resize_out_ratio
        self.camera = camera

        self._frame_sent_queue = Queue()
        self.frame_processed_queue = Queue()
        self.socket = None
        self.start_th = None
        self.sent_fps = time.time()
        self.received_fps = time.time()
        self.fps_time = time.time()

        self.res_w = 436

    def start(self, remote_server_ip):
      """
      Start the sending thread to frames to server.
      Sockets will be handled by FrameSocketStream.
      :param remote_server_id:
      :return:
      """
      self.start_th_signal = threading.Event()
      self.start_th = threading.Thread(target=self._th_start)
      if remote_server_ip is not None:
        self.remote_server = remote_server_ip

      self.start_th.start()

    def stop(self):
      try:
        if not self.start_th_signal.is_set():
          self.start_th_signal.set()
          self.start_th.join()

        # clear the queues
        with self.frame_processed_queue.mutex:
          self.frame_processed_queue.queue.clear()

        with self._frame_sent_queue.mutex:
          self._frame_sent_queue.queue.clear()

        # reset all variables
        self.sent_fps = time.time()
        self.received_fps = time.time()
        return True

      except Exception as e:
        logger.error(traceback.format_exc())
        return False
