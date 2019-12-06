import socketserver
from time import sleep

from wol.SceneNode import SceneNode

users = dict()

class ThreadedTCPRequestHandler(socketserver. BaseRequestHandler):
    def handle(self):
        global users
        killed = False
        while not killed:
            data = str(self.request.recv(1024), 'ascii')
            args = data.rstrip().split(" ")
            name = None
            if args[0] == "Hi!":
                name = args[1]
                users[name] = (0, 0, 0)
            elif args[0] == "pos":
                if name:
                    users[name] = [float(x for x in args[1:])]
            elif args[0] == "quit":
                kill = True


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


def display_positions():
    global users
    while True:
        for k, v in users:
            print(f"{k}:{v}")
        sleep(1)


class ServerNode(SceneNode):
    def __init__(self):
        SceneNode.__init__(self, name="ServerNode")

        t = threading.Thread(target=display_positions)
        t.start()

        self.port = 8971
        self.host = 'localhost'
        self.server = ThreadedTCPServer((self.host, self.port), ThreadedTCPRequestHandler)
        self.server.node = self
        self.server.context = self.context

