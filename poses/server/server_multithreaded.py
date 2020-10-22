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

n_workers = 2
print("Using 2 workers for thread pool")
futures_q = Queue(maxsize=n_workers)
worker_mgr = None
th_signal = threading.Event()
process_th = None
send_th = None
exc_info = None
exc_thrown = False

estimator = TfPoseEstimator(get_graph_path("mobilenet_thin"), target_size=(432, 368),
                            tf_config=tf.ConfigProto(log_device_placement=True))

w, h = model_wh("432x368")

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

HOST = '0.0.0.0'
PORT = 8089
conn = None
addr = None
connected = False


def wait_for_connection():
    global s, connected, conn, addr
    try:
        logger.info("Listening for connections...")
        s.listen(5)
        conn, addr = s.accept()
        conn.settimeout(30)
        logger.info("Connected to {}\nStart video processing.".format(addr))
        connected = True
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.info("Restarting...")
        wait_for_connection()
