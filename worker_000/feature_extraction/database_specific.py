import time
from helpers.influxdb_helper import InfluxDB_Helper
from influxdb.line_protocol import quote_literal, quote_ident
from feature_extraction import feature_extraction_separate
import pandas as pd

class Database_Specific(object):
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

        window = 128
        slide = 64
        fs = 50.0

        for name, group in grouped:
            if name != int(label):
                continue
            print('label:', name)
            temp_df = group.drop(['exp', 'label', 'user'], axis=1)
            
            # print(temp_df.head())
            tAcc_XYZ = temp_df.values
            all_Acc_features = feature_extraction_separate.compute_all_Acc_features(
                tAcc_XYZ, window, slide, fs)
            print(all_Acc_features.shape)
            break

        elapsed = time.time() - start
        print('elapsed:', elapsed)
        pass