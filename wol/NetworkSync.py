import threading

from PyQt5.QtGui import QVector3D

from wol.Behavior import Behavior
from wol.GeomNodes import Sphere, Avatar
import socket


class NetworkSyncToBehavior(Behavior):
    def __init__(self, obj):
        super().__init__(obj)
        s = self.obj.context.network_syncer.server_socket
        s.sendto(bytes(f"Hi! {self.obj.context.network_syncer.player_name}", 'ascii'), ('127.0.0.1', 8971))
        self.last_pos = None

    def on_update(self, dt):
        p = self.obj.position
        s = self.obj.context.network_syncer.server_socket
        if not self.last_pos == p:
            s.sendto(bytes(f"pos {p.x()} {p.y()} {p.z()}", 'ascii'), ('127.0.0.1', 8971))
            self.last_pos = QVector3D(p)


def read_server_socket(syncer):
    while syncer.running:
        s = syncer.server_socket
        s.settimeout(1.0)
        try:
            received = s.recv(1024).decode('ascii')
        except socket.timeout:
            continue
        print(received)
        arr = received.split(" ")
        print(arr)
        print(len(arr))
        if len(arr) == 5:
            cmd, name, x, y, z = arr
            if cmd == "update":
                if name == syncer.player_name:
                    continue
                if name not in syncer.players_avatars:
                    o = Avatar(name=f"avatar_{name}", parent=syncer.scene)
                    print(f"Created avatar_{name}")
                    syncer.players_avatars[name] = o
                syncer.players_avatars[name].position = QVector3D(float(x)+1, float(y), float(z))
                print(f"Updated {name}")


class NetworkSyncer:
    def __init__(self, scene):
        self.scene = scene
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.player_name = "Etaoin Shrdlu"
        self.players_avatars = dict()

        self.running = True
        self.network_thread = threading.Thread(target=read_server_socket, args=(self,))
        self.network_thread.start()
