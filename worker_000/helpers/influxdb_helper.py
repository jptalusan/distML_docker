from influxdb import InfluxDBClient, DataFrameClient
import pandas as pd

class InfluxDB_Helper:
    def __init__(self, host, port):
        self.client = InfluxDBClient(host, port)
        self.df_client = DataFrameClient(host, port)
            
    def delete_db(self, db_name):

        print("Already exists: " + db_name)
        self.client.switch_database(db_name)
        print("Switched to db: " + db_name)
        self.client.drop_database(db_name)
        print("Dropped db: " + db_name)

    def create_db(self, db_name):
        dbList = self.client.get_list_database()
        dbArr = []
        for db in dbList:
            dbArr.append(db['name'])

        if db_name in dbArr:
            print("Already exists: " + db_name)
        else:
            self.client.create_database(db_name)
            print("Created db: " + db_name)

    def write_df(self, df, db_name, meas_name, time_precision='ms', tags=None):
        self.df_client.switch_database(db_name)
        self.df_client.write_points(df, meas_name, time_precision=time_precision, tags=tags)
        
    def get_db_meas_df(self, db_name, meas_name):
    #     client = DataFrameClient(influxDB_ip, 8086, "rsu_id_location_new")
    #     result = client.query('select * from "rsu_id_location_new"."autogen"."rsu_locations";')
        self.df_client.switch_database(db_name)
        query = 'select * from "{0}"."autogen"."{1}";'.format(db_name, meas_name)
        result = self.df_client.query(query)
        return result[meas_name]
    
    def get_client(self, df=False):
        if df:
            return self.df_client
        return self.client