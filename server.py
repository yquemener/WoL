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


class SyncedObj:
    def __init__(self):
        self.pose = None
        self.last_sent_pose = None


class DataStore:
    def __init__(self):
        self.users = dict()
        self.connections = dict()
        self.connections_reverse = dict()
        self.lock = threading.Lock()

    def add_user(self, name, address, socket):
        self.lock.acquire()
        self.connections[address] = name
        self.connections_reverse[name] = (socket, address)
        self.users[name] = SyncedObj()
        self.lock.release()


datastore = DataStore()


class ThreadedUDPRequestHandler(socketserver.DatagramRequestHandler):
    def handle(self):
        global datastore
        # print("Recieved one request from {}".format(self.client_address))
        data = str(self.rfile.readline().strip(), 'ascii')
        if not data:
            print("Killed")
        # print("_"+data+"_", len(data))
        args = data.rstrip().split(" ")
        if args[0] == "Hi!":
            datastore.add_user(args[1], self.client_address,self.request[1])
        elif args[0] == "pos":
            if self.client_address in datastore.connections:
                datastore.lock.acquire()
                datastore.users[datastore.connections[self.client_address]].pose = [float(x) for x in args[1:]]
                datastore.lock.release()


class ThreadedUDPServer(socketserver.ThreadingMixIn, socketserver.UDPServer):
    pass


def display_positions():
    global datastore
    last_log = time()
    while True:
        if time() - last_log > 1.0:
            datastore.lock.acquire()
            for k, v in datastore.users.items():
                print(f"{k}:{v.pose}")
            datastore.lock.release()
            last_log = time()
        datastore.lock.acquire()
        for k, v in datastore.users.items():
            if v.last_sent_pose != v.pose:
                for s, addr in datastore.connections_reverse.values():
                    s.sendto(bytes(f"update {k} {v.pose[0]} {v.pose[1]} {v.pose[2]}", 'ascii'), addr)
                    v.last_sent_pose = v.pose
                    print(f"Sent {v.pose} to {addr}")
        datastore.lock.release()


t = threading.Thread(target=display_positions)
t.start()

port = 8971
host = 'localhost'
socketserver.UDPServer.allow_reuse_address = True
server = ThreadedUDPServer((host, port), ThreadedUDPRequestHandler)
server.serve_forever()
