import socket
import struct
from threading import Thread
from hashlib import sha256

from PyQt5.QtGui import QVector3D

from wol.CodeEdit import CodeBumperNode
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

        self.handle = Sphere(name=self.name + "_SphereHandle", parent=self)
        self.handle.scale = QVector3D(0.2, 0.2, 0.2)
        self.handle.position = QVector3D(0, 0.4, 0)
        self.handle.properties["delegateGrabToParent"] = True

        self.display_data = TextLabelNode(name=self.name + "_DataDisplayer", parent=self)
        self.display_data.properties["delegateGrabToParent"] = True

        self.publishable_objects = {self.context.current_camera.name: self.context.current_camera}
        self.subscriptions = dict()
        self.last_hash = dict()
        for k, v in self.publishable_objects.items():
            self.subscriptions[k] = set()
            hasher = sha256()
            hasher.update(v.make_pose_packet())
            self.last_hash[k] = hasher.digest()

    def __del__(self):
        self.server_running = False

    def run_server(self):
        self.server_running = True
        while self.server_running:
            data, client_address = self.socket.recvfrom(4096)
            #print(f"_{data}_")
            data = str(data, 'ascii')
            args = data.rstrip().split(" ")
            if args[0] == "Hi!" and len(args) > 1:
                name = args[1]
                self.connections[client_address] = name
                self.connections_reverse[name] = client_address
                self.users[name] = (0, 0, 0)
            elif args[0] == "pos" and len(args) > 1:
                if client_address in self.connections:
                    self.users[self.connections[client_address]] = [float(x) for x in args[1:]]

            elif args[0] == "askList":
                if client_address in self.connections:
                    publist = list()
                    for o in self.publishable_objects:
                        publist.append(o)
                    self.socket.sendto(bytes(f"PubList: {publist}", "ascii"),
                                       client_address)
            elif args[0] == "subscribe" and len(args) > 1:
                if client_address in self.connections:
                    if args[1] in self.publishable_objects:
                        self.subscriptions[args[1]].add(client_address)
                        self.socket.sendto(bytes(f"PosePacket: {args[1]} ", "ascii")
                                           + self.publishable_objects[args[1]].make_pose_packet(),
                                           client_address)

    def update(self, dt):
        s = "SERVER\n"
        s += f"Listening on {self.host}:{self.port}\n"
        s += str(self.users)+"\n"
        s += str(self.subscriptions)+"\n"
        self.display_data.set_text(s)
        for objname, subscriber_list in self.subscriptions.items():
            if len(subscriber_list) > 0:
                obj = self.publishable_objects[objname]
                hasher = sha256()
                hasher.update(obj.make_pose_packet())
                newhash = hasher.digest()
                if self.last_hash[objname] != newhash:
                    self.last_hash[objname] = newhash
                    for subscriber in subscriber_list:
                        body = bytes(f"PosePacket: {objname} ", "ascii")
                        self.socket.sendto(body + obj.make_pose_packet(), subscriber)


class ClientNode(SceneNode):
    def __init__(self, name="clientNode", parent=None, host='localhost', port=8971):
        SceneNode.__init__(self, name=name, parent=parent)

        self.port = port
        self.host = host
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.listening = True
        self.listen_thread = Thread(target=self.listen_packets, daemon=True)
        self.listen_thread.start()

        self.handle = Sphere(name=self.name + "_SphereHandle", parent=self)
        self.handle.scale = QVector3D(0.2, 0.2, 0.2)
        self.handle.position = QVector3D(0, 0.4, 0)
        self.handle.properties["delegateGrabToParent"] = True

        self.display_data = TextLabelNode(name=self.name + "_DataDisplayer", parent=self)
        self.display_data.properties["delegateGrabToParent"] = True

        self.sayHiBumper = CodeBumperNode(label="Hi", filename="pieces/SayHi", parent=self)
        self.sayHiBumper.position = QVector3D(1.0, 0, 0)

        self.sendPosBumper = CodeBumperNode(label="SendPos", filename="pieces/SendPos", parent=self)
        self.sendPosBumper.position = QVector3D(1.0, -0.1, 0)

        self.askList = CodeBumperNode(label="AskList", filename="pieces/AskList", parent=self)
        self.askList.position = QVector3D(1.0, -0.2, 0)

        self.subscribetoCam = CodeBumperNode(label="SubscribetoCam", filename="pieces/SubscribetoCam", parent=self)
        self.subscribetoCam.position = QVector3D(1.0, -0.3, 0)

        self.debugNodeParent = SceneNode(parent=self)
        self.debugNodeParent.scale = QVector3D(0.1, 0.1, 0.1)
        self.debugNode = Sphere(parent=self.debugNodeParent)

    def __del__(self):
        self.listening = False

    def update(self, dt):
        s = f"CLIENT\n"
        s += f"Sending data to {self.host}:{self.port}"
        self.display_data.set_text(s)

    def listen_packets(self):
        self.listening = True
        while self.listening:
            data, addr = self.socket.recvfrom(4096)
            args = data.split(bytes(" ", 'ascii'))
            #print(args)
            if len(args) > 2:
                if args[0] == b"PosePacket:" and args[1] == b"Camera":
                    packet = bytes(" ", 'ascii').join(args[2:])
                    self.debugNode.update_from_pose_packet(packet)
                    #p=struct.unpack("!10d", packet)
                    #pos = QVector3D(p[0], 0, 0)
                    #self.debugNode.position = pos
                    #print(pos)
                    #print(self.debugNode.world_position())


