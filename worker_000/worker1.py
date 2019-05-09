import zmq
import os
import json
import time
import pandas as pd
import multiprocessing
from feature_extraction import database_specific

current_milli_time = lambda: int(round(time.time() * 1000))

ident = os.environ['WORKER_ID']

BROKER_HOST = os.environ['BROKER_HOST']
BROKER_PORT = os.environ['BROKER_PORT']

INFLUX_HOST = os.environ['INFLUX_HOST']
INFLUX_PORT = os.environ['INFLUX_PORT']
INFLUX_DB = os.environ['INFLUX_DB']

class Worker(object):
    def __init__(self, worker_url, i):
        self.worker_url = worker_url
        self.i = i

    def parse_broker_message(self, message):
        str_request = message.decode('ascii')
        json_request = json.loads(str_request)
        print("Broker: ", json_request)

        db = database_specific.Database_Specific(INFLUX_HOST, INFLUX_PORT, INFLUX_DB)

        db.get_rows_from_db(int(json_request['label']), limit=5000)
        return json_request

    def worker_thread(self, mpq):
        """ Worker using REQ socket to do LRU routing """
        context = zmq.Context.instance()
        socket = context.socket(zmq.REQ)

        # set worker identity
        socket.identity = (u"Worker_%s" % (self.i)).encode('ascii')
        socket.connect(self.worker_url)

        # Tell the broker we are ready for work
        # Send initial stats as well, such as cpu, speed etc...
        socket.send(b"READY")

        print("Worker-{} is started...".format(ident))
        # mpq.put("I am ready and free...")
        try:
            while True:
                address, empty, request = socket.recv_multipart()
                print("I received some task...")

                mpq.put("busy")
                #  Before processing send a heartbeat
                broker_dict_task = self.parse_broker_message(request)

                print("Received in Worker %s: %s\n" % (socket.identity.decode('ascii'),
                                    request.decode('ascii')), end='')

                dict_rep = {}
                dict_rep["mean"] = "Hello"
                dict_rep["time"] = current_milli_time()
                dict_rep = json.dumps(dict_rep)

                #  Send reply back to client
                # socket.send_json(dict_rep)
                
                mpq.put("free")
                socket.send_multipart([address, b'', dict_rep.encode('ascii')])

        except zmq.ContextTerminated:
            # context terminated so quit silently
            return

class Heartbeat(object):
    def __init__(self, worker_url, i):
        self.worker_url = worker_url
        self.i = i
        self.status = "free"

    def heartbeat_process(self, mpq):
        """ Worker using REQ socket to do LRU routing """
        context = zmq.Context.instance()
        poller = zmq.Poller()
        socket = context.socket(zmq.DEALER)

        # set worker identity
        socket.identity = (u"Worker_%s" % (self.i)).encode('ascii')

        # Change socket name to "broker" or "queue"...
        socket.connect(self.worker_url)
        poller.register(socket, zmq.POLLIN)
        try:
            while True:
                # Still confused about the timeout here... 1000\ 
                socks = dict(poller.poll(1000))

                # Might not even need this receive
                # Handle worker activity on backend
                if socks.get(socket) == zmq.POLLIN:
                    frames = socket.recv_multipart()
                    # print("Receiving from heartbeat partner:", frames)
                # else:   

                while not mpq.empty():
                    self.status = mpq.get()
                    # print("MPQ:", self.status)

                # socket.send(b"Ping...")]
                lifetime = str(int(time.time()))
                # socket.send_multipart([b'', b"Ping...", self.status.encode('ascii')])
                socket.send_multipart([b'', b"Ping...", self.status.encode('ascii'), lifetime.encode('ascii')])
                # print("trying to send heartbeat...")
                time.sleep(3)

        except zmq.ContextTerminated:
            return

# https://www.geeksforgeeks.org/multiprocessing-python-set-2/

def main():
    url_worker = "tcp://{}:{}".format(BROKER_HOST, BROKER_PORT)
    url_heartb = "tcp://{}:{}".format(BROKER_HOST, 9999)
    print(url_worker, url_heartb)

    q = multiprocessing.Queue()
    
    hb = Heartbeat(url_heartb, ident)
    worker = Worker(url_worker, ident)

    multiprocessing.Process(target=worker.worker_thread, args=(q, )).start()
    multiprocessing.Process(target=hb.heartbeat_process, args=(q, )).start()

if __name__ == "__main__":
    main()