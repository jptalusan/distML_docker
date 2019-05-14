from zmq.eventloop.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream
import time
import json
from multiprocessing import Process, Queue
import queue
from collections import OrderedDict
import os
import pprint
import zmq
from lruqueue.taskcreator import TaskCreator
import random

decode = lambda x: x.decode('utf-8')
encode = lambda x: x.encode('ascii')
current_seconds_time = lambda: int(round(time.time()))

FRONTEND_PORT = os.environ['FRONTEND_PORT']
BACKEND_PORT = os.environ['BACKEND_PORT']
HEARTB_PORT = os.environ['HEARTB_PORT']

HEARTBEAT_LIVENESS = 3     # 3..5 is reasonable
HEARTBEAT_INTERVAL = float(os.environ['HEARTBEAT_INTERVAL']) #seconds

PPP_READY = (os.environ['PPP_READY']).encode('ascii')
PPP_TASKS = (os.environ['PPP_TASKS']).encode('ascii')
PPP_CAPAB_PI = (os.environ['PPP_CAPAB_PI']).encode('ascii')

PPP_FREE = (os.environ['PPP_FREE']).encode('ascii')
PPP_BUSY = (os.environ['PPP_BUSY']).encode('ascii')

PPP_HEARTBEAT = (os.environ['PPP_HEARTBEAT']).encode('ascii')

GLOBAL_TASK_LIST = []

class DistributionFactory(object):
    def __init__(self, type):
        if type == "LRU":
            dist = LeastRecentlyUsed()
        if type == "CAP":
            dist = ByCapacity()
        if type == "RND":
            dist = RandomDistribution()
        self.distributor = dist
    
    def distribute(self, backend_socket, workers, task):
        print("Distribute abstract...")
        self.distributor.distribute(backend_socket, workers, task)

class LeastRecentlyUsed(object):
    def distribute(self):
        pass
    
    def next(self):
        pass
    pass

class ByCapacity(object):
    def distribute(self):
        pass
    
    def next(self):
        pass
    pass

class RandomDistribution(object):
    def distribute(self, backend_socket, workers, task):
        print(workers)
        klist = list(workers.keys())
        print(klist)
        rlist = random.sample(klist, len(klist))
        for ind, worker in enumerate(rlist):
            rtask = task[random.randint(0, len(task) - 1)]
            backend_socket.send_multipart([encode(worker),
                                           encode(rtask)])
            task.remove(rtask)
            workers.pop(worker, None)
            print(workers)
            print(task)
    def next(self):
        pass
    pass

class Worker(object):
    def __init__(self, address, capability, status):
        if __debug__:
            print("Created worker object with address %s" % decode(address))
        self.address = decode(address)
        self.capability = decode(capability)
        self.status = decode(status)
        # self.last_alive = last_alive
        self.last_alive = current_seconds_time() + HEARTBEAT_INTERVAL * HEARTBEAT_LIVENESS

    def __repr__(self):
        ddict = {}
        ddict['capability'] = self.capability
        ddict['status'] = self.status
        ddict['last_alive'] = self.last_alive
        return json.dumps(ddict)

    def __str__(self):
        ddict = {}
        ddict['capability'] = self.capability
        ddict['status'] = self.status
        ddict['last_alive'] = self.last_alive
        return json.dumps(ddict)

class WorkerQueue(object):
    def __init__(self):
        self.queue = {}

    def ready(self, worker):
        self.queue.pop(worker.address, None)
        self.queue[worker.address] = worker

    def purge(self):
        """Look for & kill expired workers."""
        # t = time.time()
        t = current_seconds_time()
        # print("Killing expired workers at time: %s" % t)
        expired = []
        for address,worker in self.queue.items():
            # print(address, worker.last_alive)
            if t > worker.last_alive:  # Worker expired
                expired.append(address)
        for address in expired:
            print("W: Idle worker expired: %s" % address)
            self.queue.pop(address, None)

    def next(self):
        address, worker = self.queue.popitem(False)
        return address

def accept_task(task):
    # task = json.loads(task)
    # tc = TaskCreator.TaskCreator(task)
    # GLBOAL_TASK_LIST = tc.parse_request()
    # print("TASKS:{}".format(GLBOAL_TASK_LIST))

    # df = DistributionFactory("RND")
    pass

def main():
    url_client = "tcp://*:{}".format(FRONTEND_PORT)
    url_worker = "tcp://*:{}".format(BACKEND_PORT)
    url_heartb = "tcp://*:{}".format(HEARTB_PORT)

    print("Broker1 is started: %s" % (url_worker))

    context = zmq.Context()

    frontend = context.socket(zmq.ROUTER)
    frontend.bind(url_client)

    backend = context.socket(zmq.ROUTER)
    backend.bind(url_worker)

    heartbeat = context.socket(zmq.ROUTER)
    heartbeat.bind(url_heartb)

    poll_workers = zmq.Poller()
    poll_workers.register(frontend, zmq.POLLIN)
    poll_workers.register(backend, zmq.POLLIN)
    poll_workers.register(heartbeat, zmq.POLLIN)

    workers = WorkerQueue()
    heartbeat_at = time.time() + HEARTBEAT_INTERVAL

    try:
        while True:
            socks = dict(poll_workers.poll(HEARTBEAT_INTERVAL * 1000))

            if socks.get(backend) == zmq.POLLIN:
                msg = backend.recv_multipart()

                print("Received message:{}".format(msg))
                worker_addr = msg[0]

                # DEALER - [address, message]
                # Validate control message, or return reply to client
                ready, capab, stats = msg[1:4]

                workers.ready(Worker(worker_addr, capab, stats))
                if len(msg) > 4:
                    r1, r2 = msg[4:6]
                    # It means it finished some task
                    # if __debug__:
                    if GLOBAL_TASK_LIST:
                        print("Remaining tasks {:d}".format(len(GLOBAL_TASK_LIST)))
                        df = DistributionFactory("RND")
                        df.distribute(backend, workers.queue, GLOBAL_TASK_LIST)
                    else:
                        print("No tasks remaining...")
                if ready not in (PPP_READY, PPP_TASKS):
                    print("E: Invalid message from worker: %s" % msg)
                    # Accept some finished tasks from backend
                else: # Try to change to some result flag
                    # Send to frontend? or perform more tasks
                    # Check if tasks are available
                    pass
                if __debug__:
                    backend.send_multipart([worker_addr,
                        b'I have added you to the queue...',
                        b'Another message.'])

            if socks.get(heartbeat) == zmq.POLLIN:
                # Update the worker queue
                msg = heartbeat.recv_multipart()
                worker_addr = msg[0]
                ready, capab, stats = msg[1:]
                if ready == PPP_HEARTBEAT:
                    if __debug__:
                        print("Updating...")
                    workers.ready(Worker(worker_addr, capab, stats))
                if __debug__:
                    print("From HB:", msg)

            if socks.get(frontend) == zmq.POLLIN:
                print("Worker count: ", len(workers.queue))
                client_addr, _, query = frontend.recv_multipart()
                msg = [client_addr, b"", encode("HOWDY, WAIT...")]
                frontend.send_multipart(msg)
                # accept_task(query)

                tc = TaskCreator.TaskCreator(query)
                # TODO: Check this if this is correct
                GLOBAL_TASK_LIST.extend(tc.parse_request())
                print("TASKS:{}".format(GLOBAL_TASK_LIST))

                df = DistributionFactory("RND")
                
                df.distribute(backend, workers.queue, GLOBAL_TASK_LIST)
                if __debug__:
                    print(workers.queue, GLOBAL_TASK_LIST)
                pass
            else:
                # How will I attempt to send heartbeat here?
                # I think itonly triggers every second, the poll rate
                pass

            # Send heartbeats to idle workers if it's time
            if time.time() >= heartbeat_at:
                if __debug__:
                    print("Sending some heartbeats...")
                for worker in workers.queue:
                    msg = [encode(worker), PPP_HEARTBEAT]
                    heartbeat.send_multipart(msg)
                heartbeat_at = time.time() + HEARTBEAT_INTERVAL

            if __debug__:
                pprint.pprint(dict(workers.queue))
                print("Attempting to purge dead workers...")
            workers.purge()

    except zmq.ContextTerminated:
        return

if __name__ == "__main__":
    Process(target=main, args=()).start()


# Using streams instead of poller for now
# class Broker(object):
#     # Set up sockets and streams
#     def __init__(self, backend_socket):
#         self.backend = ZMQStream(backend_socket)
#         self.backend.on_recv(self.handle_backend)

#         self.loop = IOLoop.instance()

#         self.loop.make_current()

#         print("WorkerQueue:__init__()")
#         self.workers = WorkerQueue()

#     def delegate_task(self, algorithm):
#         pass

#     def handle_frontend(self):
#         pass

#     def handle_backend(self, msg):
#         print("Received message:{}".format(msg))
#         worker_addr = msg[0]

#         # DEALER - [address, message]
#         # Validate control message, or return reply to client
#         ready, capab, stats = msg[1:]
#         if ready not in (PPP_READY):
#             print("E: Invalid message from worker: %s" % msg)
#         if ready == PPP_READY:
#             self.workers.ready(Worker(worker_addr, capab, stats))
        
#         # Debug
#         pprint.pprint(dict(self.workers.queue))

#         # Check if tasks are available
#         self.backend.send_multipart([worker_addr,
#                                      b'I have added you to the queue...',
#                                      b'Another message.'])
#         pass

#     def handle_heartbeat(self):
#         pass

#     pass