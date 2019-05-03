import time
import json

current_milli_time = lambda: int(round(time.time() * 1000))

class TaskCreator(object):
    def __init__(self, client_request):
        self.client_request = client_request

    def generate_task_queue(self, json_request):
        # we get the total number of workers from heartbeat
        # or even divide by more, assuming workers can just
        # continue to work on another task after finishing one...
        task_queue = []
        time_received = current_milli_time()
        for i in range(1, 13):
            broker_req = {}
            
            broker_req["sender"] = json_request["sender"]
            broker_req["command"] = 'extract'
            broker_req["broker_received_time"] = current_milli_time()
            broker_req["broker_processed_time"] = time_received
            broker_req["dataframe"] = None
            broker_req["model"] = 'RandomForest'
            broker_req["rows"] = 100
            broker_req["label"] = i
            # broker_req["distribution"] = 1 # Make as function?

            task_queue.append(json.dumps(broker_req))

        return task_queue

    def parse_request(self):
        str_request = self.client_request.decode('ascii')
        #  Parse JSON Request
        json_request = json.loads(str_request)
        json_request = json.loads(json_request)
        print(json_request)
        print(type(json_request))

        # time_received = current_milli_time()

        # sender = json_request["sender"]
        # command = json_request["command"]
        # if json_request["dataframe"] != None:
        #     dataframe = json_request["dataframe"]

        # req_time = json_request["req_time"]
        # model = json_request["model"]
        # rows = json_request["rows"]
        # distribution = json_request["distribution"]

        # broker_req = {}

        # broker_req["sender"] = sender
        # broker_req["command"] = 'train'
        # broker_req["broker_received_time"] = current_milli_time()
        # broker_req["broker_processed_time"] = time_received
        # broker_req["dataframe"] = None
        # broker_req["model"] = 'RandomForest'
        # broker_req["rows"] = 100
        # # broker_req["distribution"] = 1 # Make as function?

        # return json.dumps(broker_req)

        return self.generate_task_queue(json_request)