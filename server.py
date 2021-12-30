"""
Implementation steps for the server:

- Show each players' position & orientation
- Keep a small list of objs in sync, sync with a Behavior
- Seed with a init_scene dump
- Sync whole scenes
- Sync behaviors
- Secure connections


"""

import socketserver
import threading
from time import sleep, time

users = dict()
connections = dict()
connections_reverse = dict()


class SyncedObj:
    def __init__(self):
        self.pose = None
        self.last_sent_pose = None


class ThreadedUDPRequestHandler(socketserver.DatagramRequestHandler):
    def handle(self):
        global users, connections, connections_reverse
        # print("Recieved one request from {}".format(self.client_address))
        data = str(self.rfile.readline().strip(), 'ascii')
        if not data:
            print("Killed")
        # print("_"+data+"_", len(data))
        args = data.rstrip().split(" ")
        if args[0] == "Hi!":
            name = args[1]
            connections[self.client_address] = name
            connections_reverse[name] = (self.request[1], self.client_address)
            users[name] = SyncedObj()
        elif args[0] == "pos":
            if self.client_address in connections:
                users[connections[self.client_address]].pose = [float(x) for x in args[1:]]


class ThreadedUDPServer(socketserver.ThreadingMixIn, socketserver.UDPServer):
    pass


def display_positions():
    global users, connections, connections_reverse
    last_log = time()
    while True:
        if time() - last_log > 1.0:
            for k, v in users.items():
                print(f"{k}:{v.pose}")
            last_log = time()
        for k, v in users.items():
            if v.last_sent_pose != v.pose:
                for s, addr in connections_reverse.values():
                    s.sendto(bytes(f"update {k} {v.pose[0]} {v.pose[1]} {v.pose[2]}", 'ascii'), addr)
                    v.last_sent_pose = v.pose
                    print(f"Sent {v.pose} to {addr}")



t = threading.Thread(target=display_positions)
t.start()

port = 8971
host = 'localhost'
socketserver.UDPServer.allow_reuse_address = True
server = ThreadedUDPServer((host, port), ThreadedUDPRequestHandler)
server.serve_forever()
