import socket
import socketserver
from threading import Thread

from PyQt5.QtGui import QVector3D

from wol.CodeBumperNode import CodeBumperNode
from wol.GeomNodes import Sphere
from wol.GuiElements import TextLabelNode
from wol.SceneNode import SceneNode


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


class ServerNode(SceneNode):
    def __init__(self, name="ServerNode", parent=None, host='localhost', port=8971):
        SceneNode.__init__(self, name=name, parent=parent)

        self.port = port
        self.host = host
        self.server = ThreadedUDPServer((self.host, self.port), ThreadedUDPRequestHandler)
        self.server.node = self
        self.server.context = self.context
        self.server.users = dict()
        self.server.connections = dict()
        self.server.connections_reverse = dict()

        self.handle = Sphere(name=self.name+"_SphereHandle", parent=self)
        self.handle.scale = QVector3D(0.2, 0.2, 0.2)
        self.handle.position = QVector3D(0, 0.4, 0)
        self.handle.properties["delegateGrabToParent"] = True

        self.display_data = TextLabelNode(name=self.name+"_DataDisplayer", parent=self)
        self.display_data.properties["delegateGrabToParent"] = True

        self.server_running = True
        self.server_thread = Thread(target=self.run_server)
        self.server_thread.start()

    def __del__(self):
        self.server_running=False

    def run_server(self):
        self.server_running = True
        while self.server_running:
            self.server.handle_request()

    def update(self, dt):
        s = "SERVER\n"
        s += f"Listening on {self.host}:{self.port}\n"
        s += str(self.server.users)
        self.display_data.set_text(s)


class ClientNode(SceneNode):
    def __init__(self, name="clientNode", parent=None, host='localhost', port=8971):
        SceneNode.__init__(self, name=name, parent=parent)

        self.port = port
        self.host = host
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.handle = Sphere(name=self.name+"_SphereHandle", parent=self)
        self.handle.scale = QVector3D(0.2, 0.2, 0.2)
        self.handle.position = QVector3D(0, 0.4, 0)
        self.handle.properties["delegateGrabToParent"] = True

        self.display_data = TextLabelNode(name=self.name+"_DataDisplayer", parent=self)
        self.display_data.properties["delegateGrabToParent"] = True

        self.sayHiBumper = CodeBumperNode(text="Hi", filename="pieces/SayHi", parent=self)
        self.sayHiBumper.position = QVector3D(1.0, 0, 0)

        self.sendPosBumper = CodeBumperNode(text="SendPos", filename="pieces/SendPos", parent=self)
        self.sendPosBumper.position = QVector3D(1.0, -0.2, 0)

    def update(self, dt):
        s = f"CLIENT\n"
        s += f"Sending data to {self.host}:{self.port}"
        self.display_data.set_text(s)
