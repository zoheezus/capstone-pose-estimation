import logging
import threading
import traceback
from queue import Empty

import cv2
from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, send, emit
import time

from poses.poseapp.poseapp_sockets import PoseAppWSockets

# app will run without aws and docker-machine.
# instead, app will run locally
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

    if not processing_started:
        try:
            poseapp.start()
            processing_started = True
        except Exception as e:
            logger.error(traceback.format_exc())
            return Response("internal server error: try again", status=500)
        return "started processing"
    else:
        return "already started"


@app.route('/stop', methods=['POST'])
def stop():
    global processing_started
    try:
        processing_started = False
        poseapp.stop()
    except Exception as e:
        logging.error(traceback.format_exc())
        return Response("unable to stop. error occurred.", status=500)
    return "stopped"


@app.route('/camera_feed')
def camera_feed():
    def gen_feed():
        fps_time = time.time()
        # while True and not poseapp.start_th_signal.is_set():
        while True:
            try:
                frame = poseapp.frame_processed_queue.get(
                    block=True, timeout=2)
                ret, jpeg = cv2.imencode('.jpg', frame)
                fps_time = time.time()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            except Empty:
                continue

        logger.info('Videostream closed')

    return Response(gen_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')
