import logging
import threading
import traceback
from queue import Empty

import cv2
from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, send, emit

from poses.poseapp.poseapp_sockets import PoseAppWSockets

app = Flask(__name__, template_folder="static")
socketio = SocketIO(app)
processing_started = False
logger = logging.getLogger(__name__)

poseapp = PoseAppWSockets(delay_time=160)

@app.route('/')
def index():
    # JSX if time permits
    return Response(render_template("index.html"))

@app.route('/start', methods=['POST'])
def start():
    # start processings
    global poseapp
    global processing_started