import socket
import time

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('127.0.0.1', 8971))
data=""
s.settimeout(1.0)
s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
while data!=b"quit":
    try:
        data, addr = s.recvfrom(4096)
        print(data)
    except socket.timeout:
        time.sleep(0.1)
        pass
