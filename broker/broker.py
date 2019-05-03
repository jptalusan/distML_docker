# import zmq

# # Prepare our context and sockets
# context = zmq.Context()
# frontend = context.socket(zmq.ROUTER)
# backend = context.socket(zmq.ROUTER)
# frontend.bind("tcp://*:7000")
# backend.bind("tcp://*:6000")

# # Initialize main loop state
# count = 3
# workers = []
# poller = zmq.Poller()
# # Only poll for requests from backend until workers are available
# poller.register(backend, zmq.POLLIN)

# print("Broker is started...")

# # Switch messages between sockets
# while True:
#     sockets = dict(poller.poll())

#     if backend in sockets:
#         # Handle worker activity on the backend
#         request = backend.recv_multipart()
#         worker, empty, client = request[:3]
#         if not workers:
#             # Poll for clients now that a worker is available
#             poller.register(frontend, zmq.POLLIN)
#         if client == b"READY":
#             print("Received READY signal from {} BE".format(worker.decode("utf-8")))
#         workers.append(worker)
#         if client != b"READY" and len(request) > 3:
#             # If client reply, send rest back to frontend
#             empty, reply = request[3:]
#             print("From BE: ", reply)
#             frontend.send_multipart([client, b"", reply])
#             # count -= 1
#             # if not count:
#             #     break

#     if frontend in sockets:
#         # Get next client request, route to last-used worker
#         client, empty, request = frontend.recv_multipart()
#         print("From Client-{} FE: ".format(client.decode("utf-8") ), request)
#         worker = workers.pop(0)
#         backend.send_multipart([worker, b"", client, b"", request])
#         if not workers:
#             # Don't poll clients if no workers are available
#             poller.unregister(frontend)

# # print("Closing broker...")

# # # Clean up
# # backend.close()
# # frontend.close()
# # context.term()
