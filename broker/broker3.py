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
import numpy as np
from itertools import repeat

import sklearn
from sklearn.ensemble import RandomForestClassifier

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

DISTRIBUTED = os.environ['DISTRIBUTED']
CENTRALIZED = os.environ['CENTRALIZED']

GLOBAL_TASK_LIST = []
secondary_queries = []
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
        if type == "RND-CENTRAL":
            dist = RandomDistributionCentralized()
        self.distributor = dist
    
    def distribute(self, backend_socket, workers, task, extra, method):
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
        self.distributor.distribute(backend_socket, workers, task, extra, method)

    def last_worker(self):
        return self.distributor.last_worker

    def distributor(self):
        return self.distributor

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

class RandomDistributionCentralized(object):
    def distribute(self, backend_socket, workers, task, extra, method=None):
        workers.pop(self.selected_worker, None)

        # TODO: FUCK SOBRANG JANKY!
        if isinstance(extra, (list)):
            message = [encode(self.selected_worker), 
                    encode(task)]
            message.extend(extra)
            backend_socket.send_multipart(message)
        elif isinstance(extra, (bytes)):
            message = [encode(self.selected_worker), 
                    encode(task),
                    extra]
            backend_socket.send_multipart(message)
        self.last_worker = self.selected_worker

    def next(self):
        pass
    
    def last_worker(self):
        return self.last_worker

    # TODO: Fix for the other function using this...
    def select_worker(self, workers, worker_address=None):
        if worker_address is not None:
            self.selected_worker = worker_address
        else:
            print("Keys:", workers.keys())
            self.selected_worker = list(workers.keys())[0]

# TODO: Just create a class that sends data to worker (since it is streamed by another machine/client...)
class RandomDistributionStreamedData(object):
    pass

# Should I multiprocess this? But the tasklist is only a single item, it might cause problems...
class RandomDistribution(object):
    def distribute(self, backend_socket, workers, tasks, extra, method):
        if __debug__:
            print("RandomDistribution:{}".format(tasks))

        for ind, _ in enumerate(tasks):
            if len(workers) == 0:
                break
            klist = list(workers.keys())
            rlist = random.sample(klist, len(klist))

            worker = rlist[ind % len(rlist)]
            rtask = tasks[random.randint(0, len(tasks) - 1)]
            global tasks_sent
            tasks_sent += 1
            if extra == None:
                backend_socket.send_multipart([encode(worker),
                                               encode(rtask)])
                print("Query {} sent to {} at {}".format(tasks_sent, worker, str(current_seconds_time())))
            else:
                # TODO: Must differentiate between centralized and distributed here as well
                # Distributed
                if method == DISTRIBUTED:
                    message = [encode(worker), 
                            encode(rtask),
                            extra[ind]]
                    print("D Query {} with extra sent to {} at {}".format(tasks_sent, worker, str(current_seconds_time())))
                    # backend_socket.send_multipart(message)
                    extra.remove(extra[ind])
                    # print("Len of extra remaining:{}".format(len(extra)))

                # Centralized
                elif method == CENTRALIZED:
                    message = [encode(worker), 
                            encode(rtask)]
                    message.extend(extra)
                    print("C Query {} with extra sent to {} at {}".format(tasks_sent, worker, str(current_seconds_time())))
                    
                elif method == 'AGGREGATE_MODELS':
                    pass

                # print("Len of extra remaining:{}".format(len(extra)))
                backend_socket.send_multipart(message)
                
            tasks.remove(rtask)
            self.last_worker = worker
            workers.pop(worker, None)
            # print("Len of tasks remaining:{}".format(len(tasks)))

    def last_worker(self):
        return self.last_worker

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

# TODO: Change name of this
def generate_2ndry_tasks(n_arr):
    pickle_arr = []
    for n_ in n_arr:
        temp = zip_and_pickle(n_)
        pickle_arr.append(temp)
    return pickle_arr

# TODO: For some reason, the shuffle is not working? I thought it was inplace.
# This might cause too much memory use though. (most probably)
def split(a, n):
    # np.random.shuffle(a)
    a_ = a[np.random.permutation(a.shape[0])]
    k, m = divmod(len(a_), n)
    return (a_[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def split_aggregated_feature_extracted(aggregated_pickles):
    output = []
    for pickled in aggregated_pickles:
        unzipped = blosc.decompress(pickled)
        unpickld = pickle.loads(unzipped)
        temp = unpickld.tolist()
        output.extend(temp)
    np_output = np.asarray(output)
    return np_output

def parse_broker_message(message):
    str_request = decode(message)
    json_request = json.loads(str_request)
    return json_request

def zip_and_pickle(obj, flags=0, protocol=-1):
    """pickle an object, and zip the pickle before sending it"""
    p = pickle.dumps(obj, protocol)
    z = blosc.compress(p, typesize=8)
    return z

def unpickle_and_unzip(pickled):
    unzipped = blosc.decompress(pickled)
    unpickld = pickle.loads(unzipped)
    return unpickld

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

                    # TODO: Just testing something, but may add for future the last flag (new)
                    # if flag not in (PPP_READY, PPP_TASKS, PPP_XTRCT, PPP_TRAIN, PPP_CLSFY):
                    if flag not in (PPP_READY, PPP_TASKS, PPP_XTRCT, PPP_TRAIN, PPP_CLSFY, "CLASSIFY_STREAM_ONLY"):
                        print("E: Invalid flag from worker: %s" % flag)
                    else: # Try to change to some result flag
                        # Send to frontend? or perform more tasks
                        pass

                    if flag == PPP_XTRCT:
                        resp1, resp2 = msg[5:7]
                        pickled = msg[8]

                        extraction_tasks_processed_by_worker += 1

                        if __debug__:
                            unzipped = blosc.decompress(pickled)
                            unpickld = pickle.loads(unzipped)

                            print("Unpickled shape: {}".format(unpickld.shape))
                        aggregated_pickles.append(pickled)
                        
                        curr_time = str(current_seconds_time())
                        print("Fextract response received from {} at {}".format(worker_addr, curr_time))

                        # DEBUGGER
                        # if __debug__:
                        reply = [client_addr, 
                                b"XTRACT_RESP",
                                worker_addr, 
                                encode(curr_time)]
                        frontend.send_multipart(reply)

                        # TODO: Idea, maybe add some task ID to verify that the task
                        # sent is the one that has been processed.
                        if GLOBAL_TASK_LIST:
                            # If the task list has not been exhausted...
                            df = DistributionFactory("RND")
                            df.distribute(backend, workers.queue, GLOBAL_TASK_LIST, extra=None, method=None)
                        else:
                            if extraction_tasks_processed_by_worker == tasks_received:
                                print("No tasks remaining...")
                                print("Next task:{}".format(parsed_query))

                                pickled_clf_arr = []
                                dist_model_accs = []
                                training_tasks_processed_by_worker = 0
                                # Distribute next tasks (train or clsasify)
                                df = DistributionFactory("RND")

                                # TODO: This is stupid hahaha
                                # TODO: This is a global variable now
                                dict_req = json.loads(parsed_query)
                                dict_req['req_time'] = current_seconds_time()

                                # Should include [distributed] as a CONSTANTS
                                if dict_req["train_dist_method"] == DISTRIBUTED:
                                    print("Im trying something here...")
                                    n_ = split_aggregated_feature_extracted(aggregated_pickles)
                                    print(n_.shape)
                                    
                                    # TODO: The split should be prepared by request or algorithm
                                    # number_of_trainers = len(workers.queue)
                                    number_of_trainers = 6

                                    n_arr = list(split(n_, number_of_trainers))
            
                                    # checking if it was randomized
                                    if np.equal(n_[0:50], n_arr[0][0:50]).all():
                                        print("Equal")
                                    else:
                                        print("Not equal")

                                    split_pickles_arr = generate_2ndry_tasks(n_arr)
                                    secondary_queries = [parsed_query]
                                    secondary_queries = [x for item in secondary_queries for x in repeat(item, len(split_pickles_arr))]
                                    # TODO: duplicate parsed_query... same as length of n_arr

                                    if __debug__:
                                        print("Len of split pickles arr: {}".format(len(split_pickles_arr)))
                                        print("Len of secondary_queries: {}".format(len(secondary_queries)))
                                        
                                    df.distribute(backend, 
                                              workers.queue, 
                                              secondary_queries, 
                                              split_pickles_arr,
                                              method=dict_req["train_dist_method"])
                                elif dict_req["train_dist_method"] == CENTRALIZED:
                                    print("training centralized")                                
                                    # TODO: Super janky, but pickle array should be same length as parsed query.                            
                                    parsed_query = json.dumps(dict_req)
                                    df.distribute(backend, 
                                                workers.queue, 
                                                [parsed_query], 
                                                aggregated_pickles,
                                                method=dict_req["train_dist_method"])

                                ''' Reset all things related to distributed worker tasks '''
                                # TODO: Dangerous, might not be successfully sent...
                                tasks_sent = 0
                                
                    if flag == PPP_TRAIN:
                        response = msg[6]
                        time_done = msg[7]
                        # TODO: WHATFHADAWD is this?>!eqeq
                        # It all relies on what the first query of the client is...
                        # WHAT IF MULTIPLE CLIENTs with DIFFERENT queries send at the same time!?
                        # dict_req = json.loads(parsed_query)

                        print("Flag {} executed with response: {}".format(flag, response))

                        if dict_req["train_dist_method"] == DISTRIBUTED:
                            training_tasks_processed_by_worker += 1
                            model_accuracy = msg[8]
                            zipped_pickled_model = msg[9]
                            
                            # TODO: Get the ML models here for aggregation.
                            # Put into array and send into a single centralized compiler (not trainer)
                            
                            pickled_clf_arr.append(zipped_pickled_model)
                            dist_model_accs.append(decode(model_accuracy))

                            # TODO: Store array of accuracies as well?
                            if secondary_queries:
                                df = DistributionFactory("RND")
                                df.distribute(backend, 
                                            workers.queue, 
                                            secondary_queries, 
                                            split_pickles_arr,
                                            method=dict_req["train_dist_method"])
                            else:
                                # TODO: Sobrang janky nito kadire
                                # TODO: Need to clear some variables...
                                if training_tasks_processed_by_worker == number_of_trainers:
                                    print("Just need to aggregate here...")
                                    number_of_trainers = 0
                                    print("Len of pickled arr: {}".format(len(pickled_clf_arr)))

                                    # TODO: Generate one more task to aggregate the models.
                                    # TODO: Need to test them
                                    dict_req = {}
                                    dict_req["sender"] = decode(client_addr)
                                    dict_req["command"] = "AGGREGATE_MODELS"
                                    dict_req["req_time"] = current_seconds_time()
                                    dict_req["model"] = 'RandomForest'
                                    dict_req["train_dist_method"] = CENTRALIZED
                                    tertiary_query = json.dumps(dict_req)
                                    df = DistributionFactory("RND")
                                    df.distribute(backend, 
                                            workers.queue, 
                                            [tertiary_query], 
                                            pickled_clf_arr,
                                            method=dict_req["train_dist_method"])

                                    # This last worker contains the aggregated ML model
                                    last_worker = df.last_worker()
                                    print("last worker:{}".format(df.last_worker()))
                                    str_model_accs = (", ").join(dist_model_accs)
                                    print(str_model_accs)
                                    print(type(str_model_accs))
                                    # TODO: Inform Client so it can just trigger classification if needed
                                    reply = [client_addr,
                                            b"TRAIN_RESP_DONE",
                                            encode(last_worker),
                                            time_done,
                                            encode(str_model_accs)]
                                    frontend.send_multipart(reply)

                                    # ~~~~~~~~~~~~~~~ # solely for validation purposes
                                    print("Dist aggregated pickles len: {}".format(len(aggregated_pickles)))

                                    # feat_extd_data = split_aggregated_feature_extracted(aggregated_pickles)
                                    # print("Dist aggregated pickle[0] shape: {}".format(feat_extd_data.shape))

                                    dict_req = {}
                                    dict_req["sender"] = decode(client_addr)
                                    dict_req["command"] = PPP_CLSFY
                                    dict_req["req_time"] = current_seconds_time()
                                    dict_req["model"] = 'RandomForest'
                                    classify_query = json.dumps(dict_req)

                                    df = DistributionFactory("RND-CENTRAL")
                                    df.distributor.select_worker(workers.queue, last_worker)
                                    # df.distributor.select_worker(workers.queue, "Worker-000")
                                    df.distribute(backend, 
                                            workers.queue, 
                                            classify_query, #CHANGE
                                            aggregated_pickles,
                                            method=None)
                                    aggregated_pickles = []

                        elif dict_req["train_dist_method"] == CENTRALIZED:
                            zipped_pickled_model = msg[9]
                            aggregated_pickles = []

                            unzipped = blosc.decompress(zipped_pickled_model)
                            clf = pickle.loads(unzipped)

                            print("Done in {}".format(decode(time_done)))
                            print("C Clf:{}".format(clf))
                            reply = [client_addr,
                                    b"TRAIN_RESP",
                                    worker_addr,
                                    time_done,
                                    msg[8]]
                            frontend.send_multipart(reply)

                    if flag == PPP_CLSFY:
                        message = decode(msg[6])
                        time_finished = decode(msg[7])
                        model_accuracy = decode(msg[8])
                        predictions = msg[9]

                        unzipped_predictions = unpickle_and_unzip(predictions)
                        print("Received from worker: {} on {} with acc: {}".format(message, time_finished, model_accuracy))
                        print(unzipped_predictions)

                        reply = [client_addr,
                                b"CLSFY_RESP",
                                worker_addr,
                                msg[7],
                                msg[8],
                                predictions]
                        frontend.send_multipart(reply)

                        # TODO4: Just resending a heartbeat for debugging/testing/experiment
                        for kk in range(12):
                            worker_name = "Worker_" + "{}".format(kk).zfill(4)
                            print("Sending a heartbeat to {}".format(worker_name))
                            msg = [encode(worker_name), PPP_HEARTBEAT]
                            heartbeat.send_multipart(msg)

                        print(workers.queue)
                        # TODO4 (END)

                    if flag == "CLASSIFY_STREAM_ONLY":
                        print("Returned classify stream")
                        message = decode(msg[6])
                        time_finished = decode(msg[7])
                        predictions = msg[8]
                        print("Client:{}".format(client_addr))
                        print("time:{}".format(time_finished))
                        print("Preds:{}".format(unpickle_and_unzip(predictions)))
                        reply = [b"Client-000",#client_addr,
                        # reply = [client_addr,
                                b"CLASSIFY_STREAM_ONLY_RESP",
                                worker_addr,
                                msg[7],
                                predictions]
                        frontend.send_multipart(reply)

            if socks.get(heartbeat) == zmq.POLLIN:
                # Update the worker queue
                msg = heartbeat.recv_multipart()
                worker_addr = msg[0]
                # print("Received heartbeat from {}".format(decode(worker_addr)))
                ready, capab, stats = msg[1:]
                if ready == PPP_HEARTBEAT:
                    if __debug__:
                        print("Updating...")
                    workers.ready(Worker(worker_addr, capab, stats))
                if __debug__:
                    print("From HB:", msg)

                if __debug__:
                    print("HB:{}".format(workers.queue.keys()))
                # print("Current workers:{}".format(workers.queue))

            # TODO: Specify here a flag/branch for when classify is triggered by the client
            ''' 
                - must check if the workers have a model (future task)
                - raw data must come from the client in chunks (what about queueing?)
                - data must be feature extracted and classified in chunks as well
                - metrics from query, collection(?), processing and aggregation
                - separate workers for classifying and training... (depending on speeds)
                - required number of chunks for features to be extracted -> 128 (window size)
                - ask nakamura, how label is set? -> labels.txt (chunks here equivalent to label on .txt file)
                - if model is found, just extract then classify (continuously?)..., else extract -> train -> classify
                - (future) while classifying, also train and see if accuracy will increase????
            '''
            if socks.get(frontend) == zmq.POLLIN:
                aggregated_pickles = []
                GLOBAL_TASK_LIST = []
                # TODO: In task creator, return a value that shows how many 
                # of a particular tasks where received, ie. CLSFY, EXTRCT etc...
                extraction_tasks_processed_by_worker = 0

                print("Worker count: ", len(workers.queue))
                # IF client is REQ
                # client_addr, _, query = frontend.recv_multipart()
                # IF client is DEALER
                # client_addr, query = frontend.recv_multipart()

                query = frontend.recv_multipart()
                client_addr = query[0]

                json_req = json.loads(decode(query[1]))
                print("Query type: {}".format(type(query)))
                # print("Query: {}".format(query))
                print(len(query))
                print(json_req)
                print(type(json_req))

                # ~~~~~~~~~ CLASSIFY ONLY ~~~~~~~~~~~~~~ #
                # TODO: Must add which 'acc' or 'gyro' is needed...
                # Hard coded worker - since im lazy (for now) centralized ML classification
                if json_req['command'] == PPP_CLSFY:
                    print("Received some classification task...")
                    pickled_chunk = query[2:]

                    dict_req = {}
                    dict_req["sender"] = decode(client_addr)
                    dict_req["command"] = "EXTRACT-CLASSIFY"
                    dict_req["req_time"] = current_seconds_time()
                    dict_req["model"] = 'RandomForest'
                    dict_req["database"] = json_req["database"]
                    classify_query = json.dumps(dict_req)
                    
                    print("Classify query: {}".format(classify_query))
                    # TODO: When no workers are available, either queue incoming data (which might be too much)
                    # or drop it, then measured the percentages of dropped vs success/sent
                    if len(workers.queue) > 0:
                        df = DistributionFactory("RND-CENTRAL")
                        df.distributor.select_worker(workers.queue, worker_address=None)
                        df.distribute(backend, 
                                workers.queue, 
                                classify_query,
                                extra=pickled_chunk,
                                method=None)
                    else:
                        print("Dropped streamed data since cannot accomodate...")
                # ~~~~~~~~~~~~~~~~~~~~~~~ #

                elif json_req['command'] == PPP_TRAIN:
                    # if json_req['command'] == "TRAIN"
                    # For generating secondary tasks (either classify, train or classify then train)
                    # parsed_query = accept_task(client_addr, query)
                    
                    # For generating tasks for all workers (initial task)
                    tc = TaskCreator.TaskCreator(json_req)

                    # TODO: Refactor, global task list is extract task...
                    tasks_array = tc.parse_request()

                    GLOBAL_TASK_LIST.extend(tasks_array["EXTRACT"])

                    # TODO: Refactor, parsed query is secondary task (train)
                    parsed_query = tasks_array["TRAIN"]

                    # TODO5: JUst trying to limit the workers here for automation

                    print(workers.queue)
                    req_workers = json_req["workers"]
                    print("Req workers:{}".format(req_workers))
                    for index, key in enumerate(list(workers.queue)):
                        if len(workers.queue) == int(req_workers):
                            print("Done clearing some workers...")
                            break
                        else:
                            del workers.queue[key]
                            ("Deleted: {}".format(key))

                    print(workers.queue)

                    # TODO5 (END)

                    print("Secondary task:{}".format(parsed_query))

                    tasks_received = len(GLOBAL_TASK_LIST)

                    if __debug__:
                        print("TASKS:{}".format(GLOBAL_TASK_LIST))

                    df = DistributionFactory("RND")
                    df.distribute(backend, workers.queue, GLOBAL_TASK_LIST, extra=None, method=None)

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
                print("Attempting to purge dead workers...")
                pprint.pprint(dict(workers.queue))
            workers.purge()
    except zmq.ContextTerminated:
        return

if __name__ == "__main__":
    Process(target=main, args=()).start()