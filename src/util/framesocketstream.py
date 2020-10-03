import logging
import socket
import threading
import threading
import traceback
import time
import struct
from queue import Queue

import pickle

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FrameSocketStream():
    def __init__(self, serverip, port):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._serverip = serverip
        self._port = port

        # Async thread to receive responses from server
        self.th_recv = None
        self.socket_is_closed = False

    @property
    def socket(self):
        return self._socket

    def init_connection(self):
        while True:
            try:
                self.socket.connect((self._serverip, int(self._port)))
                print("Connect to {}:{}".format(self._serverip, self._port))
                time.sleep(1)
                self.socket_is_closed = False
                break
            except Exception as e:
                logger.error("Exception caught {}".format(
                    traceback.format_exc()))
                raise RuntimeError("Problem connecting to server")

    def start_recv_thread(self, recv_callback):
        self.th_recv_signal = threading.Event()
        self.th_recv = threading.Thread(
            target=self._th_recv, args=(recv_callback,))
        self.th_recv.daemon = True
        self.th_recv.start()
