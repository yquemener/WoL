import socket
from time import sleep

import message_pb2 as protocol

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

s.connect(('127.0.0.1', 8971))

while True:
    msg = protocol.Message()
    msg.auth.CopyFrom(protocol.Auth())
    msg.auth.name = "Poy"
    print(f"Sending {msg}")
    s.sendall(msg.SerializeToString())
    sleep(1)