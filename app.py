import threading
from queue import Empty

from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, send, emit

app = Flask(__name__, template_folder="static")
socketio = SocketIO(app)
processing_started = False

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