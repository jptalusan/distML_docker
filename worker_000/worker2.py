import zmq
import os
import json
import time
import pandas as pd
import multiprocessing
# from feature_extraction import database_specific

decode = lambda x: x.decode('utf-8')
encode = lambda x: x.encode('ascii')
current_milli_time = lambda: int(round(time.time() * 1000))

ident = os.environ['WORKER_ID']

BROKER_HOST = os.environ['BROKER_HOST']
BROKER_PORT = os.environ['BROKER_PORT']

INFLUX_HOST = os.environ['INFLUX_HOST']
INFLUX_PORT = os.environ['INFLUX_PORT']
INFLUX_DB = os.environ['INFLUX_DB']

HEARTB_PORT = os.environ['HEARTB_PORT']

PPP_READY = (os.environ['PPP_READY']).encode('ascii')
PPP_TASKS = (os.environ['PPP_TASKS']).encode('ascii')
PPP_CAPAB_PI = (os.environ['PPP_CAPAB_PI']).encode('ascii')

PPP_FREE = (os.environ['PPP_FREE']).encode('ascii')
PPP_BUSY = (os.environ['PPP_BUSY']).encode('ascii')

PPP_HEARTBEAT = (os.environ['PPP_HEARTBEAT']).encode('ascii')
HEARTBEAT_INTERVAL = float(os.environ['HEARTBEAT_INTERVAL']) #seconds

# Must create separate process since tasks may range from instant to too long
class Heartbeat(object):
    def __init__(self, worker_url, i):
        self.worker_url = worker_url
        self.i = i
        self.identity = (u"Worker_%s" % (self.i)).encode('ascii')

    def worker_thread(self, mpq):
        """ Worker using REQ socket to do LRU routing """
        context = zmq.Context.instance()
        socket = context.socket(zmq.DEALER)

        # set worker identity
        socket.identity = self.identity
        socket.connect(self.worker_url)

        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)

        try:
            while True:
                socks = dict(poller.poll(HEARTBEAT_INTERVAL * 1000))

                if socks.get(socket) == zmq.POLLIN:
                    frames = socket.recv_multipart()
                    if __debug__:
                        print("HB: received some task...", frames)
                    socket.send_multipart([PPP_HEARTBEAT,
                                            PPP_CAPAB_PI,
                                            PPP_FREE])

        except zmq.ContextTerminated:
            # context terminated so quit silently
            return

class Worker(object):
    def __init__(self, worker_url, i):
        self.worker_url = worker_url
        self.i = i
        self.identity = (u"Worker_%s" % (self.i)).encode('ascii')

    def worker_thread(self, mpq):
        """ Worker using REQ socket to do LRU routing """
        context = zmq.Context.instance()
        socket = context.socket(zmq.DEALER)

        # set worker identity
        socket.identity = self.identity
        socket.connect(self.worker_url)

        # Tell the broker we are ready for work
        # Send initial stats as well, such as cpu, speed etc...
        # socket.send(PPP_READY)
        socket.send_multipart([PPP_READY,
                                PPP_CAPAB_PI,
                                PPP_FREE])

        try:
            while True:
                message_from_router = socket.recv_multipart()
                print("WR: received some task...", message_from_router)

                time.sleep(5)
                print("WR: done working...")
                socket.send_multipart([PPP_TASKS,
                                      PPP_CAPAB_PI,
                                      PPP_FREE,
                                      encode("Done working..."),
                                      b"Some other data..."])

        except zmq.ContextTerminated:
            # context terminated so quit silently
            return
        print("Sent ready message...")

def main():
    url_worker = "tcp://{}:{}".format(BROKER_HOST, BROKER_PORT)
    url_heartb = "tcp://{}:{}".format(BROKER_HOST, HEARTB_PORT)
    
    q = multiprocessing.Queue()
    
    worker = Worker(url_worker, ident)
    heartb = Heartbeat(url_heartb, ident)

    multiprocessing.Process(target=worker.worker_thread, args=(q, )).start()
    multiprocessing.Process(target=heartb.worker_thread, args=(q, )).start()

if __name__ == "__main__":
    main()