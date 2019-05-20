import time
from helpers.influxdb_helper import InfluxDB_Helper
from influxdb.line_protocol import quote_literal, quote_ident
from feature_extraction import feature_extraction_separate, feature_extraction
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

    def get_query_results(self, db, label, limit=0):
        """
        Returns
        -------
        collections.defaultdict
            Usually to be used with the parse_groupby_dict function
        """
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
        return result

    def get_rows_with_label_both(self, label, limit=0):
        if __debug__:
            print('Database_specific:get_rows_with_label_both()')
        # query ='select * from "acc" WHERE "label"=1 GROUP BY "exp" LIMIT 10;'
        # other database is 'gyro'
        results_acc = self.get_query_results('acc', label, limit)
        results_gyro = self.get_query_results('gyro', label, limit)

        output = []
        for i in range(1, len(results_gyro.keys()) + 1):
            exp_no = "{:02d}".format(i)
            tempA = ('acc', (('exp', exp_no),))
            tempG = ('gyro', (('exp', exp_no),))
            
            flagA = flagG = False
            
            if tempA in results_acc.keys():
                acc_df = results_acc[tempA][['x', 'y', 'z']]
                flagA = True
            if tempG in results_gyro.keys():
                gyro_df = results_gyro[tempG][['x', 'y', 'z']]
                flagG = True
            
            if flagA and flagG:
                all_feat_np = feature_extraction.compute_all_features(acc_df.values, gyro_df.values, window, slide, fs)
                temp = all_feat_np.tolist()
                output.extend(temp)
        np_output = np.asarray(output)
        return np_output

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

        result = self.get_query_results(db, label, limit)
        label_df = self.parse_groupby_dict(result)
        label_df.index = label_df.index.tz_convert('Asia/Tokyo')

        grouped = label_df.groupby('exp')
        output = []
        for name, group in grouped:
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
                temp = n_features.tolist()
            output.extend(temp)
        np_output = np.asarray(output)
        return np_output