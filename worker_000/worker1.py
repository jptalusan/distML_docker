import zmq
import os
import json
import time
from helpers.influxdb_helper import InfluxDB_Helper
from influxdb.line_protocol import quote_literal, quote_ident
import pandas as pd
from feature_extraction import feature_extraction_separate

current_milli_time = lambda: int(round(time.time() * 1000))

ident = os.environ['WORKER_ID']

BROKER_HOST = os.environ['BROKER_HOST']
BROKER_PORT = os.environ['BROKER_PORT']

INFLUX_HOST = os.environ['INFLUX_HOST']
INFLUX_PORT = os.environ['INFLUX_PORT']
INFLUX_DB = os.environ['INFLUX_DB']

def parse_groupby_dict(data, add_measurement=False):
    df_list = []

    for key, df in data.items():
        measurement, (tags) = key
        if add_measurement:
            df['measurement'] = measurement

        for tag in tags:
            name, value = tag
            df[name] = value

        df_list.append(df)
        
    return pd.concat(df_list)

def down_sample_majority_class():
    pass

def get_rows_from_db(label):
    activities = range(1, 13)

    start = time.time()
    influxdb_helper = InfluxDB_Helper(INFLUX_HOST, INFLUX_PORT)
    df_client = influxdb_helper.get_client(df=True)
    df_client.switch_database(INFLUX_DB)

    query = "SELECT * FROM {} " \
        "GROUP BY \"user\" LIMIT 50".format(quote_ident('acc'))

    result = df_client.query(query)
    label_df = parse_groupby_dict(result)
    label_df.index = label_df.index.tz_convert('Asia/Tokyo')
    print(label_df.shape)

    grouped = label_df.groupby('label')

    window = 128
    slide = 64
    fs = 50.0
    for name, group in grouped:
        if name != int(label):
            continue
        print('label:', name)
        temp_df = group.drop(['exp', 'label', 'user'], axis=1)
        
        print(temp_df.head())
        tAcc_XYZ = temp_df.values
        all_Acc_features = feature_extraction_separate.compute_all_Acc_features(
            tAcc_XYZ, window, slide, fs)
        print(all_Acc_features.shape)
        break

    elapsed = time.time() - start
    print('elapsed:', elapsed)
    pass

def parse_broker_message(message):
    str_request = message.decode('ascii')
    #  Parse JSON Request (I dont know why i need to do this twice)
    json_request = json.loads(str_request)
    print("Broker: ", json_request)

    get_rows_from_db(int(json_request['label']))
    return json_request

def worker_thread(worker_url, i):
    """ Worker using REQ socket to do LRU routing """
    context = zmq.Context.instance()

    socket = context.socket(zmq.REQ)

    # set worker identity
    socket.identity = (u"Worker-%s" % (i)).encode('ascii')

    socket.connect(worker_url)

    # Tell the broker we are ready for work
    socket.send(b"READY")

    print("Worker-{} is started...".format(ident))
    
    # make this asynchronous so I can still send heartbeat messages during working
    # Or go through the same path as Nakamura and just stop sending heartbeat so 
    # This worker will be removed from the cluster...
    try:
        while True:

            address, empty, request = socket.recv_multipart()

            #  Before processing send a heartbeat
            broker_dict_task = parse_broker_message(request)

            print("Received in Worker %s: %s\n" % (socket.identity.decode('ascii'),
                                request.decode('ascii')), end='')

            socket.send_multipart([address, b'', b'OK'])

    except zmq.ContextTerminated:
        # context terminated so quit silently
        return

def main():
    url_worker = "tcp://{}:{}".format(BROKER_HOST, BROKER_PORT)
    print(url_worker)
    worker_thread(url_worker, ident)

if __name__ == "__main__":
    main()