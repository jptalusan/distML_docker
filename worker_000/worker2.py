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

import psutil
import functools
from joblib import dump, load
from feature_extraction import feature_extraction_separate, feature_extraction

decode = lambda x: x.decode('utf-8')
encode = lambda x: x.encode('ascii')
current_seconds_time = lambda: int(round(time.time()))

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

DISTRIBUTED = os.environ['DISTRIBUTED']
CENTRALIZED = os.environ['CENTRALIZED']

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
                    
                    # print("Received HB")
                    if __debug__:
                        print(psutil.cpu_percent(interval=None, percpu=False))
                        print(psutil.cpu_percent(interval=None, percpu=True))
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

def unpickle_and_unzip(pickled):
    unzipped = blosc.decompress(pickled)
    unpickld = pickle.loads(unzipped)
    return unpickld
    
def extract_features_from_raw_data(meas, raw_np_arr, raw_np_arr_2=None):
    window = 128
    slide = 64
    fs = 50.0

    if meas == 'acc':
        tAcc_XYZ = raw_np_arr
        n_features = feature_extraction_separate.compute_all_Acc_features(
            tAcc_XYZ, window, slide, fs)
        temp = n_features.tolist()
    elif meas == 'gyro':
        tGyr_XYZ = raw_np_arr
        n_features = feature_extraction_separate.compute_all_Gyr_features(
            tGyr_XYZ, window, slide, fs)
        temp = n_features.tolist()
    elif meas == 'both':
        tAcc_XYZ = raw_np_arr
        tGyr_XYZ = raw_np_arr_2
        all_feat_np = feature_extraction.compute_all_features(
            tAcc_XYZ, tGyr_XYZ, 
            window, slide, fs)
        temp = all_feat_np.tolist()
    return np.asarray(temp)

def extract_features(label, database, limit):
    dbs = database_specific.Database_Specific(INFLUX_HOST, INFLUX_PORT, INFLUX_DB)
    if database == 'both':
        nout = dbs.get_rows_with_label_both(int(label), limit=limit)
        # print("Nout {}:{}".format(database, nout.shape))
        return nout
    elif database == 'acc' or database == 'gyro':
        nout = dbs.get_rows_with_label(int(label), database, limit=limit)
        # print("Nout {}:{}".format(database, nout.shape))
        return nout
    else:
        return None

def parse_broker_message(message):
    str_request = decode(message)
    json_request = json.loads(str_request)
    return json_request

class TrainingFactory(object):
    def __init__(self, type):
        if type == "RF":
            trainer = RandomForest()
        self.trainer = trainer
    
    def train(self, numpy_arr):
        self.trainer.train(numpy_arr)

    def accuracy(self):
        return self.trainer.accuracy()

    def zipped_pickle_model(self):
        return zip_and_pickle(self.trainer.model())

class RandomForest(object):
    def train(self, numpy_arr):
        print("RandomForest()")
        X = numpy_arr[:,:-1]
        y = numpy_arr[:,-1:]

        if __debug__:
            print(X.shape)
            print(y.shape)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

        if __debug__:
            print(X_train.shape)
            print(y_train.shape)
            print(X_test.shape)
            print(y_test.shape)

        # sc = StandardScaler()
        # # https://stackoverflow.com/questions/48692500/fit-transform-on-training-data-and-transform-on-test-data
        # X_train = sc.fit_transform(X_train)
        # X_test = sc.transform(X_test)

        self.classifier = RandomForestClassifier(n_estimators=20, random_state=100)  
        self.classifier.fit(X_train, y_train.ravel())  
        y_pred = self.classifier.predict(X_test)  

        print(confusion_matrix(y_test,y_pred))
        print(classification_report(y_test,y_pred))
        self.accuracy_score = accuracy_score(y_test, y_pred)
        print("accuracy:{}".format(self.accuracy_score))

    def accuracy(self):
        return self.accuracy_score

    def next(self):
        pass
    
    def model(self):
        return self.classifier

    def predict(self, numpy_arr):
        pass

    def validate(self, X, y):

        pass

    pass

class Classifier(object):
    pass

def split_aggregated_feature_extracted(aggregated_pickles):
    output = []
    for pickled in aggregated_pickles:
        unzipped = blosc.decompress(pickled)
        unpickld = pickle.loads(unzipped)
        temp = unpickld.tolist()
        output.extend(temp)
    np_output = np.asarray(output)
    return np_output

def combine_rfs(rf_a, rf_b):
    rf_a.estimators_ += rf_b.estimators_
    rf_a.n_estimators = len(rf_a.estimators_)
    return rf_a

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
                # print("WR: received some tasklen:{}".format(len(message_from_router)))
            
                json_req = parse_broker_message(message_from_router[0])
                client_addr = json_req['sender']
                command = json_req['command']
                if command == PPP_XTRCT:
                    print("Query received from broker at {}".format(str(current_seconds_time())))

                    label = json_req['label']
                    database = json_req['database']
                    rows = int(json_req['rows'])
                    # DEBUG
                    numpy_arr = extract_features(label, database, rows)
                    if numpy_arr.size != 0:
                        label_col = np.full((numpy_arr.shape[0], 1), int(label))
                        numpy_arr = np.append(numpy_arr, label_col, axis=1)
                        zipped = zip_and_pickle(numpy_arr)
                        print("Query processed by {} in {}".format(self.identity, str(current_seconds_time())))
                        # print("After label: ", numpy_arr.shape)
                        # print("WR: done working...")

                        # print("After sending:", psutil.cpu_percent(interval=None, percpu=False))
                        print("After sending:", psutil.cpu_percent(interval=None, percpu=True))
                        socket.send_multipart([PPP_TASKS,
                                            PPP_CAPAB_PI,
                                            PPP_FREE,
                                            encode(PPP_XTRCT),
                                            encode(client_addr),
                                            encode("Done working..."),
                                            b"Some other data...",
                                            zipped])
                    else:
                        print("Failed feature extraction. Chunk too small (<128)")
                elif command == PPP_TRAIN:
                    print(json_req['command'])
                    pickle_arr = message_from_router[1:]

                    print("number of pickles received: {}".format(len(pickle_arr)))
                    np_output = split_aggregated_feature_extracted(pickle_arr)
                    print(np_output.shape)

                    # TODO: Should I clear this each time? a new message arrives?
                    tf = TrainingFactory("RF")
                    tf.train(np_output)
                    
                    socket.send_multipart([PPP_TASKS,
                                        PPP_CAPAB_PI,
                                        PPP_FREE,
                                        encode(PPP_TRAIN),
                                        encode(client_addr),
                                        b"Im done with TRAINING...",
                                        encode(str(current_seconds_time())),
                                        encode(str(tf.accuracy())),
                                        tf.zipped_pickle_model()])
                # TODO: This is specifically for distributedly trained RF!
                elif command == "AGGREGATE_MODELS":
                    print(json_req['command'])
                    pickle_arr = message_from_router[1:]

                    unpickled_arr = []
                    for ind, pickled in enumerate(pickle_arr):
                        unzipped = blosc.decompress(pickled)
                        clf = pickle.loads(unzipped)
                        print("C Clf{}:{}".format(ind, clf))
                        unpickled_arr.append(clf)

                    combined_rf_model = functools.reduce(combine_rfs, unpickled_arr)

                    # TODO: I guess ok for now. But if i only have a single volume, this is weird/wrong
                    dump(combined_rf_model, 'combined_rf_model.joblib') 

                    # TODO: Test or classify to verify effect
                    print("Combined Clf:{}".format(combined_rf_model))
                    pass

                elif command == PPP_CLSFY:
                    print("Got something from here...")
                    pickle_arr = message_from_router[1:]
                    np_output = split_aggregated_feature_extracted(pickle_arr)
                    print("Pickle array for validation...", np_output.shape)

                    # ~~~~~~~~~~~~~~# DEBUGGING
                    X = np_output[:,:-1]
                    y = np_output[:,-1:]

                    X_train, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


                    # TODO: It fails without this, how do I do machine learning?!
                    # sc = StandardScaler()
                    # X_train = sc.fit_transform(X_train)
                    # X_test = sc.transform(X_test)

                    clf = load('combined_rf_model.joblib') 
                    y_pred = clf.predict(X_test)

                    print(confusion_matrix(y_test, y_pred))
                    print(classification_report(y_test, y_pred))
                    acc = accuracy_score(y_test, y_pred)
                    print("accuracy:{}".format(acc))
                    print("final CLF: {}".format(clf))

                    pickled_y_pred = zip_and_pickle(y_pred)

                    socket.send_multipart([PPP_TASKS,
                                           PPP_CAPAB_PI,
                                           PPP_FREE,
                                           encode(PPP_CLSFY),
                                           encode(client_addr),
                                           b"Im done with Classifying...",
                                           encode(str(current_seconds_time())),
                                           encode(str(acc)),
                                           pickled_y_pred])
                    # ~~~~~~~~~~~~~~~# END
                    pass

                elif command == "EXTRACT-CLASSIFY":
                    print("EXTRACT-CLASSIFY TASK received!")
                    pickled_arr = message_from_router[1:]

                    meas = json_req["database"]

                    # TODO: WOW HARDCODED
                    if meas == 'both':
                        np_arr = unpickle_and_unzip(pickled_arr[0])
                        np_arr2 = unpickle_and_unzip(pickled_arr[1])
                        output = extract_features_from_raw_data(meas, np_arr, raw_np_arr_2=np_arr2)
                    else:
                        np_arr = unpickle_and_unzip(pickled_arr[0])
                        output = extract_features_from_raw_data(meas, np_arr)

                    # TODO: Anyway to check if joblib load is still loaded?
                    clf = load('combined_rf_model.joblib')
                    y_pred = clf.predict(output)
                    pickled_y_pred = zip_and_pickle(y_pred)
                    
                    print("Extracted:{}".format(output.shape))
                    socket.send_multipart([PPP_TASKS,
                                           PPP_CAPAB_PI,
                                           PPP_FREE,
                                           encode("CLASSIFY_STREAM_ONLY"),
                                           encode(client_addr),
                                           b"Im done with Classifying...",
                                           encode(str(current_seconds_time())),
                                           pickled_y_pred])

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