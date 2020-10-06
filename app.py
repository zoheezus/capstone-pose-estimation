import logging
import threading
import traceback
from queue import Empty

import cv2
from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, send, emit

from src.poseapp.poseapp_sockets import PoseAppWSockets

app = Flask(__name__, template_folder="static")
socketio = SocketIO(app)
processing_started = False
logger = logging.getLogger(__name__)

poseapp = PoseAppWSockets(delay_time=160)

@app.route

@app.route('/')
def index():
    # index.html needs to be created
    # JSX if time permits
    return Response(render_template("index.html"))

@app.route('/start', methods=['POST'])
def start():
    # start processings
    # delete pass when writing code
    pass