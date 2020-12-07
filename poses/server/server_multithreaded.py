import logging
import traceback
import time

from poses.poseapp.poseapp_sockets import PoseAppWSockets
from tf_pose_estimation.tf_pose.estimator import TfPoseEstimator
from tf_pose_estimation.tf_pose.networks import model_wh, get_graph_path

import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

estimator = TfPoseEstimator(get_graph_path("mobilenet_thin"), target_size=(432, 368),
                            tf_config=tf.ConfigProto(log_device_placement=True))
w, h = model_wh("432x368")


def _worker_th(frame):
    global estimator, w, h
    humans = estimator.inference(frame, resize_to_default=(
        w > 0 and h > 0), upsample_size=4.0)
    pose = "none"
    if len(humans) > 0:
        humans.sort(key=lambda x: x.score, reverse=True)
        # get the human with the highest score
        humans = humans[:1]
        frame = TfPoseEstimator.draw_humans(frame, humans)
        frame, pose = PoseAppWSockets.indentify_body_gestures(frame, humans[0])

    return frame, pose


if __name__ == "__main__":
    fps_time = time.time()

    while True:
        try:
            _worker_th()

            fps_time = time.time()
        except Exception as e:
            logger.error("Exception caught! {}".format(
                traceback.format_exc()))
            continue
