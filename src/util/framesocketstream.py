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

    def _th_recv(self, callback):
      data = b''
      payload_size = struct.calcsize("<L")
      while True:
        try:
          while len(data) < payload_size:
            data += self.socket.recv(8192)

          logger.debug("Frame size received. Size: {}\n".format(len(data)))
          packed_msg_size = data[:payload_size]
          msg_payload_size = struct.unpack("<L", packed_msg_size)[0]

          # Get human data
          data = data[payload_size:]
          while len(data) < msg_payload_size:
            data += self.socket.recv(8192)

          logger.debug("Frame received. Size: {}\n".format(len(data)))
        except socket.timeout:
          # auto close socket
          logger.info("Socket timeout at recv thread. Continuing...")
          continue

        except Exception:
          logger.error("Exception occured: {}\n".format(traceback.format_exc()))
          break

        msg = data[:msg_payload_size]
        # check if server is sending close signal
        if msg == b'close':
          logger.info("received closed from server")
          if not self.socket_is_closed:
            try:
              self.socket.shutdown(2)
              self.socket.close()
              logger.infor("Socket closed")
              self.socket_is_closed = True

            except Exception as e:
              logger.error("Error closing socket. {}".format(traceback))
          break

        try:
          data = data[msg_payload_size:]
          # convert the frame and human data back to its original form
          frame, pose = pickle.loads(msg)
          callback(frame, pose)
        except Exception as e:
          logger.error(traceback.format_exc())
          continue

      logger.info("Exiting receiving thread")