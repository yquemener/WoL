import threading

from PyQt5.QtGui import QVector3D, QQuaternion

from wol.Behavior import Behavior
from wol.GeomNodes import Sphere, Avatar
import socket
import wol.protobuf.message_pb2 as protocol

class NetworkSyncToBehavior(Behavior):
    def __init__(self, obj):
        super().__init__(obj)
        self.last_pos = None
        self.last_orient = None

    def on_update(self, dt):
        p = self.obj.position
        o = self.obj.orientation
        s = self.obj.context.network_syncer.server_socket
        if not self.last_pos == p:
            s.sendto(bytes(f"pos {p.x()} {p.y()} {p.z()}", 'ascii'), ('127.0.0.1', 8971))
            self.last_pos = QVector3D(p)
        if not self.last_orient == o:
            s.sendto(bytes(f"orient {o.x()} {o.y()} {o.z()} {o.scalar()}", 'ascii'), ('127.0.0.1', 8971))
            self.last_orient = QQuaternion(o)


def read_server_socket(syncer):
    while syncer.running:
        s = syncer.server_socket
        s.settimeout(1.0)
        try:
            received = s.recv(1024).decode('ascii')
        except socket.timeout:
            continue
        arr = received.split(" ")
        cmd = arr[0]
        print(arr)
        if cmd == "update":
            name = arr[2]
            if name == syncer.player_name:
                continue
            if name not in syncer.players_avatars:
                o = Avatar(name=f"avatar_{name}", parent=syncer.scene)
                print(f"Created avatar_{name}")
                syncer.players_avatars[name] = o

            field = arr[1]
            if field == "pos":
                _, _, _, x, y, z = arr
                syncer.players_avatars[name].position = QVector3D(float(x), float(y), float(z))
            if field == "orient":
                _, _, _, ox, oy, oz, ow = arr
                syncer.players_avatars[name].orientation = QQuaternion(float(ow), float(ox), float(oy), float(oz))
            print(f"Updated {name}")


class NetworkSyncer:
    def __init__(self, scene):
        self.scene = scene
        self.server_socket = None
        self.player_name = "Etaoin Shrdlu"
        self.players_avatars = dict()
        self.running = False
        self.network_thread = False

    def connect(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.running = True
        self.network_thread = threading.Thread(target=read_server_socket, args=(self,))
        self.network_thread.start()
        self.server_socket.sendto(
            bytes(f"knockknock {self.player_name}", 'ascii'), ('127.0.0.1', 8971))
