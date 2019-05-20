import time
import json
import os

current_seconds_time = lambda: int(round(time.time()))
PPP_XTRCT = os.environ['PPP_XTRCT']
PPP_TRAIN = os.environ['PPP_TRAIN']
PPP_CLSFY = os.environ['PPP_CLSFY']

class TaskCreator(object):
    def __init__(self, client_request):
        self.client_request = client_request

    def generate_task_queue(self, json_request):
        # we get the total number of workers from heartbeat
        # or even dividgreenubi
        # e by more, assuming workers can just
        # continue to work on another task after finishing one...
        task_queue = []
        time_received = current_seconds_time()
        # Range is based on number of classes
        # TODO: Really need to change this
        for i in range(1, 13):
            broker_req = {}
            
            broker_req["sender"] = json_request["sender"]
            broker_req["command"] = PPP_XTRCT
            broker_req["broker_received_time"] = current_seconds_time()
            broker_req["broker_processed_time"] = time_received
            broker_req["database"] = json_request["database"]
            broker_req["model"] = json_request["model"]
            broker_req["rows"] = json_request["rows"]
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
        # print(type(json_request))

        return self.generate_task_queue(json_request)