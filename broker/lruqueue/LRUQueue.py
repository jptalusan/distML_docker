from zmq.eventloop.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream
from .constants import NBR_WORKERS, NBR_CLIENTS
import time
import json
from .taskcreator import TaskCreator
from multiprocessing import Process, Queue
import queue

current_milli_time = lambda: int(round(time.time() * 1000))

class LRUQueue(object):
    """LRUQueue class using ZMQStream/IOLoop for event dispatching"""

    def __init__(self, backend_socket, frontend_socket):
        self.available_workers = 0
        self.workers = []
        self.task_queue = []
        self.client_nbr = NBR_CLIENTS

        self.backend = ZMQStream(backend_socket)
        self.frontend = ZMQStream(frontend_socket)
        self.backend.on_recv(self.handle_backend)

        self.loop = IOLoop.instance()
        self.mpq = Queue()

    def handle_backend(self, msg):
        # Queue worker address for LRU routing
        worker_addr, empty, client_addr = msg[:3]

        assert self.available_workers < NBR_WORKERS

        # add worker back to the list of workers
        self.available_workers += 1
        print('Added to available workers (new total): ', self.available_workers)
        self.workers.append(worker_addr)

        #   Second frame is empty
        assert empty == b""

        # Third frame is READY or else a client reply address
        if client_addr == b'READY':
            print("Received READY signal from {} BE".format(worker_addr.decode("utf-8")))

        # If client reply, send rest back to frontend
        if client_addr != b"READY":
            empty, reply = msg[3:]

            # Following frame is empty
            assert empty == b""

            print("Received task, done by {}, to be sent to {}".format(
                worker_addr.decode("utf-8"), 
                client_addr.decode("utf-8")))

            # Receive something from backend and if the same length as the task queue
            # Aggregate and then send a response to the client
            # 
            self.frontend.send_multipart([client_addr, b'', reply])

            self.client_nbr -= 1

            print('Client count: ', self.client_nbr)

            # commented out because don't want to close broker immediately
            # if self.client_nbr == 0:
            #     # Exit after N messages
            #     print('Exiting...')
            #     self.loop.add_timeout(time.time() + 1, self.loop.stop)

        # Should change this? Or I should break down the data 
        # received on the frontend to multiple tasks for different workers.
        # a queue of tasks.
        if self.available_workers == 1:
            # on first recv, start accepting frontend messages
            print("Accepting client messages now...")
            try:
                print("Status of queue: ", self.mpq.get(True, 0.1))
            except queue.Empty:
                print("No queue yet...")

            self.frontend.on_recv(self.handle_frontend)
            # If a worker is available, pop a task from the queue.

    def process_task_queue(self, q, client_addr, task_list):
        self.mpq.put('Task list len: ', len(task_list))
        print("Process task queue started.")
        while len(task_list) > 0:
            for task in task_list:
                print(task)
                if self.available_workers > 0:
                    worker_id = self.workers.pop()
                    self.available_workers -= 1

                    self.backend.send_multipart([worker_id, 
                                                b'', 
                                                client_addr, 
                                                b'', 
                                                task.encode('ascii')])
                    task_list.remove(task)

    def handle_frontend(self, msg):
        # Now get next client request, route to LRU worker
        # Client request is [address][empty][request]
        client_addr, empty, request = msg
        print(client_addr)
        assert empty == b""

        tc = TaskCreator.TaskCreator(request)
        self.task_queue = tc.parse_request()
        
        print("Task Queue:", self.task_queue)

        # p = Process(target=self.process_task_queue, args=(self.mpq, client_addr, task_queue, ))
        # p.start()

        # if self.available_workers == 3:
        #     for task in task_queue:
        #         worker_id = self.workers.pop()
        #         self.available_workers -= 1
        #         self.backend.send_multipart([worker_id, 
        #                                      b'', 
        #                                      client_addr, 
        #                                      b'', 
        #                                      task.encode('ascii')])

        # print("Request:", json.loads(broker_request))

        #  Perform some algorithms here based on task creator
        # request = "Hello".encode('ascii')
        #  Or create the queue here? and send to multiple workers. We wait until
        #  a worker is available and then we 
        #  Dequeue and drop the next worker address
        # self.available_workers -= 1
        # worker_id = self.workers.pop()

        # print('Worker count: ', self.available_workers)
        # # self.backend.send_multipart([worker_id, b'', client_addr, b'', request])
        # self.backend.send_multipart([worker_id, 
        #                              b'', 
        #                              client_addr, 
        #                              b'', 
        #                              task_queue.encode('ascii')])

        if self.available_workers == 0:
            print('Stopping reception until workers are available.')
            # stop receiving until workers become available again
            self.frontend.stop_on_recv()