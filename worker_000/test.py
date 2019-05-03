import time
import zmq
import os
import zlib
import pickle
import blosc
import json
import pandas as pd

from feature_extraction import acc_feature_extraction
from feature_extraction import feature_extraction

pd.set_option('display.max_colwidth', -1)
pd.options.display.float_format = '{:.16g}'.format

current_milli_time = lambda: int(round(time.time() * 1000))

def recv_zipped_pickle(socket, flags=0, protocol=-1):
    """inverse of send_zipped_pickle"""
    z = socket.recv(flags)
    p = blosc.decompress(z)
    return pickle.loads(p)

HOST = '*'
PORT = '6000'

# ================dont edit================#
_context = zmq.Context()
socket = _context.socket(zmq.REP)
url = 'tcp://{}:{}'.format(HOST, PORT)
print(url)
print("HELLO")
# socket.bind(url)
socket.bind('tcp://*:6000')

while True:
    #  Wait for next request from client
    df = recv_zipped_pickle(socket)
    print(df.shape)
    mean = df.mean(axis=0)


    #  Do some 'work'
    # time.sleep(1)
    tAcc_XYZ = df.values
    window = 128
    slide = 64
    fs =50.0
    all_Acc_features = acc_feature_extraction.compute_all_Acc_features(tAcc_XYZ, window, slide, fs)
    print(all_Acc_features.shape)

    dict_rep = {}
    dict_rep["mean"] = mean.to_json()
    dict_rep["time"] = current_milli_time()
    dict_rep = json.dumps(dict_rep)

    #  Send reply back to client
    socket.send_json(dict_rep)