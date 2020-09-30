import threading
from queue import Empty

from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, send, emit

app = Flask(__name__, template_folder="static")
socketio = SocketIO(app)
processing_started = False

