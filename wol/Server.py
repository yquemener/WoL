import socket
from threading import Thread

from PyQt5.QtGui import QVector3D

from wol.CodeBumperNode import CodeBumperNode
from wol.GeomNodes import Sphere
from wol.GuiElements import TextLabelNode
from wol.SceneNode import SceneNode


class ServerNode(SceneNode):
    def __init__(self, name="ServerNode", parent=None, host='localhost', port=8971):
        SceneNode.__init__(self, name=name, parent=parent)

        self.port = port
        self.host = host
        self.users = dict()
        self.connections = dict()
        self.connections_reverse = dict()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.host, self.port))
        self.server_running = True
        self.server_thread = Thread(target=self.run_server, daemon=True)
        self.server_thread.start()

        self.handle = Sphere(name=self.name+"_SphereHandle", parent=self)
        self.handle.scale = QVector3D(0.2, 0.2, 0.2)
        self.handle.position = QVector3D(0, 0.4, 0)
        self.handle.properties["delegateGrabToParent"] = True

        self.display_data = TextLabelNode(name=self.name+"_DataDisplayer", parent=self)
        self.display_data.properties["delegateGrabToParent"] = True

    def __del__(self):
        self.server_running=False

    def run_server(self):
        self.server_running = True
        while self.server_running:
            data, client_address = self.socket.recvfrom(4096)
            print(f"_{data}_")
            data = str(data, 'ascii')
            args = data.rstrip().split(" ")
            if args[0] == "Hi!":
                name = args[1]
                self.connections[client_address] = name
                self.connections_reverse[name] = client_address
                self.users[name] = (0, 0, 0)
            elif args[0] == "pos":
                if client_address in self.connections:
                    self.users[self.connections[client_address]] = [float(x) for x in args[1:]]
            elif args[0] == "askList":
                print("ask list")
                print(client_address)
                if client_address in self.connections:
                    self.socket.sendto(bytes(f"['Cam']", "ascii"),
                                                       client_address)
                    pass

    def update(self, dt):
        s = "SERVER\n"
        s += f"Listening on {self.host}:{self.port}\n"
        s += str(self.users)
        self.display_data.set_text(s)


class ClientNode(SceneNode):
    def __init__(self, name="clientNode", parent=None, host='localhost', port=8971):
        SceneNode.__init__(self, name=name, parent=parent)

        self.port = port
        self.host = host
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.listening = True
        self.listen_thread = Thread(target=self.listen_packets, daemon=True)
        self.listen_thread.start()

        self.handle = Sphere(name=self.name+"_SphereHandle", parent=self)
        self.handle.scale = QVector3D(0.2, 0.2, 0.2)
        self.handle.position = QVector3D(0, 0.4, 0)
        self.handle.properties["delegateGrabToParent"] = True

        self.display_data = TextLabelNode(name=self.name+"_DataDisplayer", parent=self)
        self.display_data.properties["delegateGrabToParent"] = True

        self.sayHiBumper = CodeBumperNode(text="Hi", filename="pieces/SayHi", parent=self)
        self.sayHiBumper.position = QVector3D(1.0, 0, 0)

        self.sendPosBumper = CodeBumperNode(text="SendPos", filename="pieces/SendPos", parent=self)
        self.sendPosBumper.position = QVector3D(1.0, -0.1, 0)

        self.askList = CodeBumperNode(text="AskList", filename="pieces/AskList", parent=self)
        self.askList.position = QVector3D(1.0, -0.2, 0)

        self.subscribetoCam = CodeBumperNode(text="SubscribetoCam", filename="pieces/SubscribetoCam", parent=self)
        self.subscribetoCam.position = QVector3D(1.0, -0.3, 0)

    def __del__(self):
        self.listening = False

    def update(self, dt):
        s = f"CLIENT\n"
        s += f"Sending data to {self.host}:{self.port}"
        self.display_data.set_text(s)

    def listen_packets(self):
        self.listening = True
        while self.listening:
            data = self.socket.recvfrom(4096)
            print(data)
