## Just uncomment everything
# import zmq
# import time

# NBR_CLIENTS = 10
# NBR_WORKERS = 3

# client_nbr = NBR_CLIENTS

# # Prepare our context and sockets
# context = zmq.Context()

# frontend = context.socket(zmq.ROUTER)
# frontend.bind("tcp://*:7000")

# backend = context.socket(zmq.ROUTER)
# backend.bind("tcp://*:6000")

# # Initialize main loop state
# count = 3
# workers = []

# # Queue of available workers
# available_workers = 0
# workers_list = []

# poller = zmq.Poller()

# # Always poll for worker activity on backend
# poller.register(backend, zmq.POLLIN)

# # Poll front-end only if we have available workers
# poller.register(frontend, zmq.POLLIN)

# print("Broker1 is started...")

# # Switch messages between sockets
# while True:
#     socks = dict(poller.poll())

#     if (backend in socks and socks[backend] == zmq.POLLIN):
#         # Queue worker address for LRU routing
#         message = backend.recv_multipart()
#         assert available_workers < NBR_WORKERS

#         worker_addr = message[0]

#         # add worker back to the list of workers
#         available_workers += 1
#         workers_list.append(worker_addr)

#         #   Second frame is empty
#         empty = message[1]
#         assert empty == b""

#         # Third frame is READY or else a client reply address
#         client_addr = message[2]
#         if client_addr == b'READY':
#             print("Received READY signal from {} BE".format(worker_addr.decode("utf-8")))

#         # If client reply, send rest back to frontend
#         if client_addr != b'READY':
#             # Following frame is empty
#             empty = message[3]
#             assert empty == b""

#             reply = message[4]
#             print("Received task done by {} to be sent to {}".format(
#                 worker_addr.decode("utf-8"), 
#                 client_addr.decode("utf-8")))

#             frontend.send_multipart([client_addr, b"", reply])

#             client_nbr -= 1

#             if client_nbr == 0:
#                 break  # Exit after N messages

#     if available_workers > 0:
#         if (frontend in socks and socks[frontend] == zmq.POLLIN):
#             # Now get next client request, route to LRU worker
#             # Client request is [address][empty][request]

#             [client_addr, empty, request] = frontend.recv_multipart()
#             assert empty == b""

#             #  Dequeue and drop the next worker address
#             available_workers += -1
#             worker_id = workers_list.pop()

#             #  This is why there is a message[4] in line 59
#             backend.send_multipart([worker_id, b"",
#                                     client_addr, b"", request])

# time.sleep(1)

# frontend.close()
# backend.close()
# context.term()