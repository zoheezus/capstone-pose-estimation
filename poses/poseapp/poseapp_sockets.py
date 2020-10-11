import logging
import threading
import traceback
from queue import Queue

import cv2
from tf_pose_estimation.tf_pose.estimator import TfPoseEstimator
from tf_pose_estimation.tf_pose.networks import get_graph_path, model_wh

from poses.util.framesocketstream import FrameSocketStream
from poses.poseapp.posegeom import PoseGeom

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

    @staticmethod
    def translate_to_actual_dims(w, h, normalized_pixels_x, normalized_pixels_y):
        x, y = (int(round(w * normalized_pixels_x + 0.5)),
                int(round(h * normalized_pixels_y)))
        return x + 15, y

    def draw_frame(self, frame, pose):
        frame = self.resize_image_aspect_ratio(
            frame, width=640, inter=cv2.INTER_LINEAR)

        logger.debug("pose: {}".format(pose))
        try:
            cv2.putText(frame, "FPS %f" % (1.0 / (time.time() - self.received_fps)),
                        (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except ZeroDivisionError:
            logger.error("FPS division error")

        self.received_fps = time.time()
        self.frame_processed_queue.put(frame)

    def draw_humans(self, humans, frame):
        frame = TfPoseEstimator.draw_humans(frame, humans)

        try:
            cv2.putText(frame, "FPS: %f" % (1.0 / (time.time() - self.received_fps)),
                        (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
        except ZeroDivisionError:
            logger.error("FPS division error")

        self.received_fps = time.time()
        self.frame_processed_queue.put(frame)

    @staticmethod
    def indentify_body_gestures(frame, human):
        joint_list = human.body_parts
        pose = "none"
        fontsize = 0.5

        try:
            image_h, image_w = frame.shape[:2]

            # calculate angle between left shoulder and left elbow
            if joint_list.keys() >= {PoseGeom.LEFT_SHOULDER, PoseGeom.LEFT_ELBOW}:
                angle_2_3 = PoseGeom.angle_btw_2_points(joint_list[PoseGeom.LEFT_SHOULDER],
                                                        joint_list[PoseGeom.LEFT_ELBOW])

                cv2.putText(frame, "angle: %0.2f" % angle_2_3,
                            PoseAppWSockets.translate_to_actual_dims(image_w, image_h,
                                                                     joint_list[PoseGeom.LEFT_SHOULDER].x - 0.27,
                                                                     joint_list[PoseGeom.LEFT_SHOULDER].y),
                            cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 255, 0), 2)

            # calculate angle between left elbow and left hand
            if joint_list.keys() >= {PoseGeom.LEFT_ELBOW, PoseGeom.LEFT_HAND}:
                angle_3_4 = PoseGeom.angle_btw_2_points(joint_list[PoseGeom.LEFT_ELBOW],
                                                        joint_list[PoseGeom.LEFT_HAND])

                cv2.putText(frame, "angle: %0.2f" % angle_3_4,
                            PoseAppWSockets.translate_to_actual_dims(image_w, image_h,
                                                                     joint_list[PoseGeom.LEFT_ELBOW].x - 0.27,
                                                                     joint_list[PoseGeom.LEFT_ELBOW].y),
                            cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 255, 0), 2)

            # calculate angle between right shoulder and right elbow
            if joint_list.keys() >= {PoseGeom.RIGHT_SHOULDER, PoseGeom.RIGHT_ELBOW}:
                angle_5_6 = PoseGeom.angle_btw_2_points(joint_list[PoseGeom.RIGHT_SHOULDER],
                                                        joint_list[PoseGeom.RIGHT_ELBOW])

                cv2.putText(frame, "angle: %0.2f" % angle_5_6,
                            PoseAppWSockets.translate_to_actual_dims(image_w, image_h,
                                                                     joint_list[PoseGeom.RIGHT_SHOULDER].x,
                                                                     joint_list[PoseGeom.RIGHT_SHOULDER].y),
                            cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 255, 0), 2)

            # calculate angle between right elbow and right hand
            if joint_list.keys() >= {PoseGeom.RIGHT_ELBOW, PoseGeom.RIGHT_HAND}:
                angle_6_7 = PoseGeom.angle_btw_2_points(joint_list[PoseGeom.RIGHT_ELBOW],
                                                        joint_list[PoseGeom.LEFT_HAND])

                cv2.putText(frame, "angle: %0.2f" % angle_6_7,
                            PoseAppWSockets.translate_to_actual_dims(image_w, image_h,
                                                                     joint_list[PoseGeom.RIGHT_ELBOW].x,
                                                                     joint_list[PoseGeom.RIGHT_ELBOW].y),
                            cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 255, 0), 2)

            # calculate distance between 2 hands
            if joint_list.keys() >= {PoseGeom.LEFT_HAND, PoseGeom.RIGHT_HAND}:
                distance_4_7 = PoseGeom.distance_btw_2_points(joint_list[PoseGeom.LEFT_HAND],
                                                              joint_list[PoseGeom.RIGHT_HAND])

                cv2.putText(frame, "distance: %0.2f" % distance_4_7,
                            PoseAppWSockets.translate_to_actual_dims(image_w, image_h,
                                                                     joint_list[PoseGeom.RIGHT_HAND].x,
                                                                     joint_list[PoseGeom.RIGHT_HAND].y),
                            cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 255, 0), 2)
        except Exception as e:
            logger.error(traceback.format_exc())

        return frame, pose

    def resize_image_aspect_ratio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]
        
        if width is None and height is None:
            # return original image
            return image

        # check to see if width is None
        if width is None:
            # calculate the ratio of the height and
            # construct the dimensions
            r = height / float(h)
            dim = (int(w*r), height)

        # otherwise, height is None
        else:
            # calculate the ratio of the width and
            # construct the dimensions
            r = width / float(w)
            dim = (width, int(h*r))

        # resize image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return resized image
        return resized