from __future__ import print_function
import threading
import zmq
import random
import sys
import time
import json
import blosc
import pickle
import numpy as np

NBR_CLIENTS = 1

decode = lambda x: x.decode('utf-8')
encode = lambda x: x.encode('ascii')
current_seconds_time = lambda: int(round(time.time()))

def zip_and_pickle(obj, flags=0, protocol=-1):
    """pickle an object, and zip the pickle before sending it"""
    p = pickle.dumps(obj, protocol)
    z = blosc.compress(p, typesize=8)
    return z

def unpickle_and_unzip(pickled):
    unzipped = blosc.decompress(pickled)
    unpickld = pickle.loads(unzipped)
    return unpickld
    
def client_thread(client_url, i):
    """ Basic request-reply client using REQ socket """
    context = zmq.Context.instance()
    socket = context.socket(zmq.DEALER)

    # Set client identity. Makes tracing easier
    socket.identity = (u"Client-%s" % str(i).zfill(3)).encode('ascii')

    socket.connect(client_url)

    #  Send request, get reply
    # socket.send(b"HELLO")

    # TODO: Add option to classify instead, (assumming a model already exists)
    dict_req = {}

    dict_req["sender"] = socket.identity.decode('ascii')
    dict_req["command"] = 'TRAIN'
    dict_req["req_time"] = current_seconds_time()
    dict_req["database"] = 'acc'
    dict_req["model"] = 'RandomForest'
    dict_req["rows"] = 128
    dict_req["distribution"] = "RND"
    dict_req["train_dist_method"] = "distributed"
    # dict_req["train_dist_method"] = "centralized"

    # dict_req = json.dumps(dict_req)

    # TODO: Need to not send to just get response from classification test.
    # socket.send_json(dict_req)

    print("I am: {}".format(socket.identity))
    start = current_seconds_time()
    print("Query sent at: {}".format(start))
    try:
      while True:
        reply = socket.recv_multipart()
        print("Reply flag: {}, len: {}".format(decode(reply[0]), len(reply)))
        
        elapse_t = int(decode(reply[2])) - start
        if reply[0] == b"XTRACT_RESP":
            # print("{} finished extracting at {}".format(decode(reply[1]), decode(reply[2])))
            print("{} finished extracting in {} secs".format(decode(reply[1]), elapse_t))
        elif reply[0] == b"TRAIN_RESP":
            # print("{} finished training at {}, accuracy: {}".format(decode(reply[1]), 
            #                                                         decode(reply[2]),
            #                                                         decode(reply[3])))
            print("{} finished training in {} secs, w/ accuracy: {}".format(decode(reply[1]), 
                                                                    elapse_t,
                                                                    decode(reply[3])))
        elif reply[0] == b"TRAIN_RESP_DONE":
            print("{} finished dist training in {} secs, w/ accuracies: {}".format(decode(reply[1]), 
                                                                    elapse_t,
                                                                    decode(reply[3])))
        elif reply[0] == b"CLSFY_RESP":
            print("{} finished classification in {} secs, w/ accuracies: {}".format(decode(reply[1]), 
                                                                    elapse_t,
                                                                    decode(reply[3])))
            print("Classifications:{}".format(unpickle_and_unzip(reply[4])))

        elif reply[0] == b"CLASSIFY_STREAM_ONLY_RESP":
            print("{} classified in {} secs: [{}]".format(decode(reply[1]), 
                                                                    elapse_t,
                                                                    unpickle_and_unzip(reply[3])))
        elif len(reply) == 4:
            _, some_val, _, pickled = reply
            
            unpickld = unpickle_and_unzip(pickled)
        
            # pickle_arr = reply[4:]
            print(len(reply))
            print("Received a pickle:{}".format(unpickld.shape))
        elif len(reply) > 4:
            _, some_val, _ = reply[0:3]
            pickle_arr = reply[3:]
            print("number of pickles received: {}".format(len(pickle_arr)))
            output = []
            for pickled in pickle_arr:
                unpickld = unpickle_and_unzip(pickled)
                temp = unpickld.tolist()
                output.extend(temp)
            np_output = np.asarray(output)
            print(np_output.shape)

            print(current_seconds_time() - start)
    except zmq.ContextTerminated:
      return

def main():

  url_client = "tcp://163.221.125.55:7000"
  #context = zmq.Context()

  for i in range(NBR_CLIENTS):
    thread_c = threading.Thread(target=client_thread,
                                args=(url_client, i, ))
    #thread_c.daemon = True
    thread_c.start()
  

if __name__ == "__main__":
    main()
