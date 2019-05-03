# import zmq
# import os

# ident = os.environ['WORKER_ID']

# context = zmq.Context()
# socket = context.socket(zmq.REQ)
# socket.identity = u"Worker-{}".format(ident).encode("ascii")
# socket.connect("tcp://broker:6000")

# print("Worker-{} is started...".format(ident))
# socket.send(b"READY")

# try:
#     while True:
#         address, empty, request = socket.recv_multipart()
#         print("Received by {}: Task {} from {}".format(socket.identity.decode("ascii"),
#                                 request.decode("ascii"),
#                                 address.decode("ascii")))
#         #  Can add more using pickle or what
#         socket.send_multipart([address, b"", b"World"])
# except zmq.ContextTerminated:
#     print('Terminated')

