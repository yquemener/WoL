import socketserver

from wol.GuiElements import TextLabelNode


class ThreadedUDPRequestHandler(socketserver.DatagramRequestHandler):
    def handle(self):
        #print("Recieved one request from {}".format(self.client_address))
        data = str(self.rfile.readline().strip(), 'ascii')
        if data:
            #print("_"+data+"_", len(data))
            pass
        args = data.rstrip().split(" ")
        if args[0] == "Hi!":
            name = args[1]
            self.server.connections[self.client_address] = name
            self.server.connections_reverse[name] = self.client_address
            self.server.users[name] = (0, 0, 0)
        elif args[0] == "pos":
            if self.client_address in self.server.connections:
                self.server.users[self.server.connections[self.client_address]] = [float(x) for x in args[1:]]


class ThreadedUDPServer(socketserver.ThreadingMixIn, socketserver.UDPServer):
    pass


class ServerNode(TextLabelNode):
    def __init__(self, name="ServerNode", parent=None):
        TextLabelNode.__init__(self, name=name, parent=parent)

        self.port = 8971
        self.host = 'localhost'
        self.server = ThreadedUDPServer((self.host, self.port), ThreadedUDPRequestHandler)
        self.server.node = self
        self.server.context = self.context
        self.server.users = dict()
        self.server.connections = dict()
        self.server.connections_reverse = dict()

    def update(self, dt):
        self.set_text(str(self.server.users))
        super(ServerNode, self).update(dt)
        self.server.handle_request()
