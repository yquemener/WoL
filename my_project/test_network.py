import socket

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.sendto(bytes("quit", 'ascii'), ('127.0.0.1', 8971))