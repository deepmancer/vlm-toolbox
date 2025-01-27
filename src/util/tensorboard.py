import socket
import subprocess

from config.logging import TENSORBOARD_DEFAULT_PORT
from config.path import EXPERIMENTS_LOGGING_DIR


class TensorboardConnector:
    def __init__(self, logdir=EXPERIMENTS_LOGGING_DIR, port=TENSORBOARD_DEFAULT_PORT):
        self.logdir = logdir
        self.port = port
        self.process = None

    @staticmethod
    def check_port(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    @staticmethod
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    def start(self, logging_fn=print):
        if TensorboardConnector.check_port(self.port):
            logging_fn(f"Default port {self.port} is in use. Finding a free port...")
            self.port = TensorboardConnector.find_free_port()

        command = f"tensorboard --logdir={self.logdir} --port={self.port}"
        self.process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging_fn(f"TensorBoard started at http://localhost:{self.port}/ (PID={self.process.pid})")
        return self

    def stop(self, logging_fn=print):
        if self.process is not None:
            self.process.terminate()
            self.process.wait()
            logging_fn(f"TensorBoard process {self.process.pid} terminated.")
        else:
            logging_fn("TensorBoard is not running.")
        return self

    def __repr__(self):
        return f"TensorboardConnector(logdir={self.logdir}, port={self.port})"
