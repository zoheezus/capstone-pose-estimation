import concurrent
import logging

import os
import pickle
import socket

import struct
import threading
import traceback
from concurrent import futures
from queue import Queue, Empty

import sys
import time

import cv2
from tf_pose_estimation.tf_pose import common
from tf_pose_estimation.tf_pose.estimator import TfPoseEstimator
from tf_pose_estimation.tf_pose.networks import model_wh, get_graph_path
from poses.poseapp.poseapp_sockets import PoseAppWSockets

import tensorflow as tf

logging.basicConfig(
    stream=sys.stdout,
    format="('%(threadName)s - %(levelname)s - %(message)s",
    level=logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
