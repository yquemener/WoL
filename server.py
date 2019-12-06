import socketserver
import threading
from time import sleep

from wol.SceneNode import SceneNode

users = dict()
connections = dict()
connections_reverse = dict()


class ThreadedUDPRequestHandler(socketserver.DatagramRequestHandler):
    def handle(self):
        global users, connections, connections_reverse
        #print("Recieved one request from {}".format(self.client_address))
        data = str(self.rfile.readline().strip(), 'ascii')
        if not data:
            print("Killed")
        #print("_"+data+"_", len(data))
        args = data.rstrip().split(" ")
        if args[0] == "Hi!":
            name = args[1]
            connections[self.client_address] = name
            connections_reverse[name] = self.client_address
            users[name] = (0, 0, 0)
        elif args[0] == "pos":
            if self.client_address in connections:
                users[connections[self.client_address]] = [float(x) for x in args[1:]]


class ThreadedUDPServer(socketserver.ThreadingMixIn, socketserver.UDPServer):
    pass


def display_positions():
    global users
    while True:
        for k, v in users.items():
            print(f"{k}:{v}")
        sleep(1)


t = threading.Thread(target=display_positions)
t.start()

port = 8971
host = 'localhost'
socketserver.UDPServer.allow_reuse_address = True
server = ThreadedUDPServer((host, port), ThreadedUDPRequestHandler)
server.serve_forever()


