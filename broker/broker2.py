import zmq
import time
from lruqueue.LRUQueue import LRUQueue
from zmq.eventloop.ioloop import IOLoop
import os
from multiprocessing import Process

FRONTEND_PORT = os.environ['FRONTEND_PORT']
BACKEND_PORT = os.environ['BACKEND_PORT']

def main():
    """main method"""

    url_worker = "tcp://*:{}".format(BACKEND_PORT)
    url_client = "tcp://*:{}".format(FRONTEND_PORT)

    print("Broker1 is started...")

    # Prepare our context and sockets
    context = zmq.Context()
    frontend = context.socket(zmq.ROUTER)
    frontend.bind(url_client)
    backend = context.socket(zmq.ROUTER)
    backend.bind(url_worker)

    # create queue with the sockets
    queue = LRUQueue(backend, frontend)

    # start reactor
    IOLoop.instance().start()

if __name__ == "__main__":
    # main()
    Process(target=main, args=()).start()
