import time
from helpers.influxdb_helper import InfluxDB_Helper
from influxdb.line_protocol import quote_literal, quote_ident
from feature_extraction import feature_extraction_separate
import pandas as pd
import numpy as np

window = 128
slide = 64
fs = 50.0

class Database_Specific(object):
    """Connects to the InfluxDB and allows gathering of information and processing."""
    def __init__(self, INFLUX_HOST, INFLUX_PORT, INFLUX_DB):
        self.INFLUX_HOST = INFLUX_HOST
        self.INFLUX_PORT = INFLUX_PORT
        self.INFLUX_DB = INFLUX_DB

    def parse_groupby_dict(self, data, add_measurement=False):
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

    def down_sample_majority_class(self):
        pass

    def get_rows_with_label(self, label, db, limit=0):
        """
        Returns a numpy array of all the extracted features for all labels.
        
        Parameters
        ----------
        label : int
            A value corresponding to the class label of the data needed
        db : str
            Either 'acc' or 'gyro'
        limit : int
            Total number of rows for each label(?)

        Returns
        -------
        numpy_array
            an array of all concatenated experiments with the same label
        """
        if __debug__:
            print('Database_specific:get_rows_with_label()')
        # query ='select * from "acc" WHERE "label"=1 GROUP BY "exp" LIMIT 10;'
        # other database is 'gyro'
        grouping = 'exp'
        start = time.time()
        influxdb_helper = InfluxDB_Helper(self.INFLUX_HOST, self.INFLUX_PORT)
        df_client = influxdb_helper.get_client(df=True)
        df_client.switch_database(self.INFLUX_DB)

        query = ""
        if limit == 0:
            query = "SELECT * FROM {} " \
                    "WHERE \"label\"={} " \
                    "GROUP BY \"{}\" ".format(quote_ident(db), label, grouping)
        else:
            query = "SELECT * FROM {} " \
                    "WHERE \"label\"={} " \
                    "GROUP BY \"{}\" " \
                    "LIMIT {}".format(quote_ident(db), label, grouping, limit)
        
        result = df_client.query(query)
        label_df = self.parse_groupby_dict(result)
        label_df.index = label_df.index.tz_convert('Asia/Tokyo')
        print(label_df.shape)
        # print(label_df.head())

        grouped = label_df.groupby('exp')
        # print(grouped.head())
        output = []
        for name, group in grouped:
            # if name != int(label):
            #     continue
            # print('label:', name)
            temp_df = group.drop(['exp', 'label', 'user'], axis=1)
            
            temp = []
            if db == 'acc':
                tAcc_XYZ = temp_df.values
                n_features = feature_extraction_separate.compute_all_Acc_features(
                    tAcc_XYZ, window, slide, fs)
                temp = n_features.tolist()
            elif db == 'gyro':
                tGyr_XYZ = temp_df.values
                n_features = feature_extraction_separate.compute_all_Gyr_features(
                    tGyr_XYZ, window, slide, fs)
            # print("n_features.shape:{}".format(n_features.shape))
                temp = n_features.tolist()
            output.extend(temp)
        np_output = np.asarray(output)
        return np_output

    def get_rows_from_db(self, label, limit=0):
        start = time.time()
        influxdb_helper = InfluxDB_Helper(self.INFLUX_HOST, self.INFLUX_PORT)
        df_client = influxdb_helper.get_client(df=True)
        df_client.switch_database(self.INFLUX_DB)

        if limit == 0:
            query = "SELECT * FROM {} " \
                "GROUP BY \"user\"".format(quote_ident('acc'))
        else:
            query = "SELECT * FROM {} " \
                "GROUP BY \"user\" LIMIT {}".format(quote_ident('acc'), limit)

        result = df_client.query(query)
        label_df = self.parse_groupby_dict(result)
        label_df.index = label_df.index.tz_convert('Asia/Tokyo')
        print(label_df.shape)

        grouped = label_df.groupby('label')

        for name, group in grouped:
            if name != int(label):
                continue
            print('label:', name)
            temp_df = group.drop(['exp', 'label', 'user'], axis=1)
            
            # print(temp_df.head())
            tAcc_XYZ = temp_df.values
            all_Acc_features = feature_extraction_separate.compute_all_Acc_features(
                tAcc_XYZ, window, slide, fs)

        elapsed = time.time() - start
        print('elapsed:', elapsed)
        pass