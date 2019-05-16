import zmq
import os
import json
import time
import pandas as pd
import multiprocessing
from feature_extraction import database_specific

import numpy as np
import pickle
import zlib
import blosc

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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

PPP_XTRCT = os.environ['PPP_XTRCT']
PPP_TRAIN = os.environ['PPP_TRAIN']
PPP_CLSFY = os.environ['PPP_CLSFY']

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

def zip_and_pickle(obj, flags=0, protocol=-1):
    """pickle an object, and zip the pickle before sending it"""
    p = pickle.dumps(obj, protocol)
    z = blosc.compress(p, typesize=8)
    return z

def extract_features(label, database, limit):
    dbs = database_specific.Database_Specific(INFLUX_HOST, INFLUX_PORT, INFLUX_DB)
    # db.get_rows_from_db(int(json_request['label']), limit=5000)
    nout = dbs.get_rows_with_label(int(label), database, limit=limit)
    print("Nout:{}".format(nout.shape))
    return nout

def parse_broker_message(message):
    str_request = decode(message)
    json_request = json.loads(str_request)
    # client_addr = json_request['sender']

    # db = database_specific.Database_Specific(INFLUX_HOST, INFLUX_PORT, INFLUX_DB)
    # # db.get_rows_from_db(int(json_request['label']), limit=5000)
    # nout = db.get_rows_with_label(int(json_request['label']), 'gyro', limit=2000)
    # print("Nout:{}".format(nout.shape))
    return json_request

class TrainingFactory(object):
    def __init__(self, type):
        if type == "RF":
            trainer = RandomForest()
        self.trainer = trainer
    
    def train(self, numpy_arr):
        self.trainer.train(numpy_arr)

class RandomForest(object):
    def train(self, numpy_arr):
        print("RandomForest()")
        X = numpy_arr[:,:-1]
        y = numpy_arr[:,-1:]

        print(X.shape)
        print(y.shape)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        regressor = RandomForestClassifier(n_estimators=20, random_state=0)  
        regressor.fit(X_train, y_train)  
        y_pred = regressor.predict(X_test)  

        print(confusion_matrix(y_test,y_pred))
        print(classification_report(y_test,y_pred))
        print(accuracy_score(y_test, y_pred))

    def next(self):
        pass
    pass

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
                print("WR: received some tasklen:{}".format(len(message_from_router)))

                json_req = parse_broker_message(message_from_router[0])
                client_addr = json_req['sender']
                command = json_req['command']
                if command == PPP_XTRCT:
                    label = json_req['label']
                    # DEBUG
                    numpy_arr = extract_features(label, 'gyro', 200)
                    label_col = np.full((numpy_arr.shape[0], 1), int(label))
                    numpy_arr = np.append(numpy_arr, label_col, axis=1)
                    print("With label: ", numpy_arr.shape)
                    # numpy_arr = np.random.rand(2, 2)
                    zipped = zip_and_pickle(numpy_arr)
                    
                    # print(type(zipped))
                    # print(numpy_arr)
                    # time.sleep(1)
                    print("WR: done working...")
                    socket.send_multipart([PPP_TASKS,
                                        PPP_CAPAB_PI,
                                        PPP_FREE,
                                        encode(PPP_XTRCT),
                                        encode(client_addr),
                                        encode("Done working..."),
                                        b"Some other data...",
                                        zipped])
                elif command == PPP_TRAIN:
                    print(json_req['command'])
                    pickle_arr = message_from_router[1:]

                    print("number of pickles received: {}".format(len(pickle_arr)))
                    output = []
                    for pickled in pickle_arr:
                        unzipped = blosc.decompress(pickled)
                        unpickld = pickle.loads(unzipped)
                        temp = unpickld.tolist()
                        output.extend(temp)
                    np_output = np.asarray(output)
                    print(np_output.shape)

                    tf = TrainingFactory("RF")
                    tf.train(np_output)
                    
                    socket.send_multipart([PPP_TASKS,
                                        PPP_CAPAB_PI,
                                        PPP_FREE,
                                        encode(PPP_TRAIN),
                                        encode(client_addr),
                                        b"Im done with TRAINING..."])

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