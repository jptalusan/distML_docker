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
import blosc
import pickle

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

PPP_XTRCT = os.environ['PPP_XTRCT']
PPP_TRAIN = os.environ['PPP_TRAIN']
PPP_CLSFY = os.environ['PPP_CLSFY']

GLOBAL_TASK_LIST = []
tasks_received = 0
tasks_sent = 0
aggregated_pickles = []
parsed_query = {}

class DistributionFactory(object):
    def __init__(self, type):
        if type == "LRU":
            dist = LeastRecentlyUsed()
        if type == "CAP":
            dist = ByCapacity()
        if type == "RND":
            dist = RandomDistribution()
        self.distributor = dist
    
    def distribute(self, backend_socket, workers, task, extra):
        """
        An interface to distribute tasks to workers depending on the algorithm used
        
        Parameters
        ----------
        backend_socket : Socket
            Where messages will be sent
        workers : dict
            Dict of available Workers (found in WorkerQueue)
        task : list
            List of tasks that need to be sent to objects on the other side of the backend_socket
        extra : bytes
            Extra information (TODO: change to a list so it can be extended)
        Returns
        -------
        None
        """
        self.distributor.distribute(backend_socket, workers, task, extra)

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
    def distribute(self, backend_socket, workers, tasks, extra):
        klist = list(workers.keys())
        rlist = random.sample(klist, len(klist))
        if __debug__:
            print("RandomDistribution:{}".format(tasks))

        if extra:
            print("Extra {}".format(len(extra)))
            print("There is extra...")
        for ind, _ in enumerate(tasks):
            worker = rlist[ind % len(rlist)]
            rtask = tasks[random.randint(0, len(tasks) - 1)]
            global tasks_sent
            tasks_sent += 1
            if extra == None:
                backend_socket.send_multipart([encode(worker),
                                               encode(rtask)])
            else:
                message = [encode(worker), 
                           encode(rtask)]

                message.extend(extra)
                backend_socket.send_multipart(message)
            tasks.remove(rtask)
            workers.pop(worker, None)
            
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

def accept_task(client_addr, task):
    str_request = decode(task)
    #  Parse JSON Request
    json_request = json.loads(str_request)
    json_request = json.loads(json_request)

    command = json_request['command']
    dict_req = {}
    if command == PPP_TRAIN:

        dict_req["sender"] = decode(client_addr)
        dict_req["command"] = PPP_TRAIN
        dict_req["req_time"] = current_seconds_time()
        dict_req["model"] = 'RandomForest'

    elif command == PPP_CLSFY:
        pass
    elif command == (PPP_TRAIN + PPP_CLSFY):
        pass
    else:
        print("Incorrect task flag: {}".format(command))

    dict_req = json.dumps(dict_req)
    return dict_req

def zip_and_pickle(obj, flags=0, protocol=-1):
    """pickle an object, and zip the pickle before sending it"""
    p = pickle.dumps(obj, protocol)
    z = blosc.compress(p, typesize=8)
    return z

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
                print("Received message length:{}".format(len(msg)))
                # print("Received message:{}".format(msg))
                worker_addr = msg[0]

                # DEALER - [address, message]
                # Validate control message, or return reply to client
                ready, capab, stats = msg[1:4]

                workers.ready(Worker(worker_addr, capab, stats))
                if __debug__:
                    backend.send_multipart([worker_addr,
                        b'I have added you to the queue...',
                        b'Another message.'])

                if ready == PPP_TASKS:
                    flag = decode(msg[4])
                    client_addr = msg[5]

                    if flag not in (PPP_READY, PPP_TASKS, PPP_XTRCT, PPP_TRAIN, PPP_CLSFY):
                        print("E: Invalid flag from worker: %s" % decode(flag))
                    else: # Try to change to some result flag
                        # Send to frontend? or perform more tasks
                        pass

                    if flag == PPP_XTRCT:
                        resp1, resp2 = msg[5:7]
                        pickled = msg[8]

                        extraction_tasks_processed_by_worker += 1

                        unzipped = blosc.decompress(pickled)
                        unpickld = pickle.loads(unzipped)
                        aggregated_pickles.append(pickled)

                        # DEBUGGER
                        reply = [client_addr, 
                                b"", 
                                resp1, 
                                b"",
                                pickled]
                        frontend.send_multipart(reply)

                        # TODO: Idea, maybe add some task ID to verify that the task
                        # sent is the one that has been processed.
                        if GLOBAL_TASK_LIST:
                            # If the task list has not been exhausted...
                            df = DistributionFactory("RND")
                            df.distribute(backend, workers.queue, GLOBAL_TASK_LIST, None)
                        else:
                            if extraction_tasks_processed_by_worker == tasks_received:
                                print("No tasks remaining...")
                                print("Next task:{}".format(parsed_query))

                                # Distribute next tasks (train or clsasify)
                                df = DistributionFactory("RND")

                                # TODO: This is stupid hahaha
                                dict_req = json.loads(parsed_query)
                                dict_req['req_time'] = current_seconds_time()
                                dict_req = json.dumps(dict_req)

                                print("Am i really failing here?")
                                print(len(aggregated_pickles))

                                df.distribute(backend, 
                                              workers.queue, 
                                              [parsed_query], 
                                              aggregated_pickles)

                                ''' Reset all things related to distributed worker tasks '''
                                # TODO: Dangerous, might not be successfully sent...
                                tasks_sent = 0
                                
                    if flag == PPP_TRAIN:
                        response = msg[6]
                        aggregated_pickles = []
                        print("Flag {} executes with response: {}".format(flag, response))
                        reply = [client_addr, 
                                b"", 
                                encode("Done with training...")]
                        frontend.send_multipart(reply)
                        pass
                    if flag == PPP_CLSFY:
                        rows_classified = msg[6]
                        classifications = msg[7]
                        pass

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
                aggregated_pickles = []
                # TODO: In task creator, return a value that shows how many 
                # of a particular tasks where received, ie. CLSFY, EXTRCT etc...
                extraction_tasks_processed_by_worker = 0

                print("Worker count: ", len(workers.queue))
                # IF client is REQ
                # client_addr, _, query = frontend.recv_multipart()
                # IF client is DEALER
                client_addr, query = frontend.recv_multipart()

                # For generating secondary tasks (either classify, train or classify then train)
                parsed_query = accept_task(client_addr, query)

                print(parsed_query)

                # For generating tasks for all workers (initial task)
                tc = TaskCreator.TaskCreator(query)
                GLOBAL_TASK_LIST.extend(tc.parse_request())
                tasks_received = len(GLOBAL_TASK_LIST)

                if __debug__:
                    print("TASKS:{}".format(GLOBAL_TASK_LIST))

                df = DistributionFactory("RND")
                df.distribute(backend, workers.queue, GLOBAL_TASK_LIST, None)

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