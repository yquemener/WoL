import threading
from time import sleep

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
        self.syncer = obj.context.network_syncer
        self.network_uid = -1
        self.syncer.queue_object_creation(self)

    def on_update(self, dt):
        if self.network_uid == -1:
            return

        p = self.obj.position
        o = self.obj.orientation
        syncer = self.obj.context.network_syncer
        if not self.last_pos == p:
            syncer.queue_update_field(self.network_uid, "position",
                                      [self.obj.position.x(),
                                       self.obj.position.y(),
                                       self.obj.position.z()])
            self.last_pos = QVector3D(p)
        if not self.last_orient == o:
            syncer.queue_update_field(self.network_uid, "orientation",
                                      [self.obj.orientation.x(),
                                       self.obj.orientation.y(),
                                       self.obj.orientation.z(),
                                       self.obj.orientation.scalar()])
            self.last_orient = QQuaternion(o)


class NetworkSyncer:
    VALID_FIELDS = ["position", "orientation", "scale"]

    def __init__(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.authenticated = False
        self.player_name = "Etaoin Shrdlu"
        self.player_uid = -1
        self.players_avatars = dict()
        self.running = False
        self.listen_thread = None
        self.sending_thread = None
        self.updates_queue = list()
        self.creation_queue = list()
        self.creation_queue_waiting_answer = list()

    def connect(self):
        self.server_socket.connect(('127.0.0.1', 8971))
        self.running = True
        self.listen_thread = threading.Thread(target=self.listen)
        self.listen_thread.start()

        self.sending_thread = threading.Thread(target=self.send)
        self.sending_thread.start()

        msg = protocol.Message()
        msg.auth.CopyFrom(protocol.Auth())
        msg.auth.name = self.player_name
        self.server_socket.sendall(msg.SerializeToString())

    def listen(self):
        while self.running:
            sleep(0.016)
            s = self.server_socket
            # s.settimeout(1.0)
            try:
                received = s.recv(1024)
            except socket.timeout:
                continue
            if len(received) == 0:
                continue
            msg = protocol.Message()
            msg.ParseFromString(received)

            print(msg.WhichOneof("content"))
            if msg.WhichOneof("content") == "auth_response":
                if msg.auth_response.success:
                    self.player_uid = msg.auth_response.your_id
                    self.authenticated = True
                else:
                    print(f"Authentication failed with error: {msg.auth_response.error}")

            if msg.WhichOneof("content") == "object_state_update":
                # if msg.object_state_update.object_id == syncer.player
                for fupdate in msg.object_state_update.field:
                    if fupdate.field_name == "position":
                        print(QVector3D(fupdate.float_value[0],
                                        fupdate.float_value[1],
                                        fupdate.float_value[2]))

                    if fupdate.field_name == "orient":
                        print(QQuaternion(fupdate.float_value[0],
                                          fupdate.float_value[1],
                                          fupdate.float_value[2],
                                          fupdate.float_value[3]))

            elif msg.WhichOneof("content") == "new_object_create":
                waiter = self.creation_queue_waiting_answer.pop()
                waiter.network_uid = msg.new_object_create.object_id

    def send(self):
        while self.running:
            sleep(0.016)
            if len(self.updates_queue)>0:
                (uid, field_name, value) = self.updates_queue.pop()
                msg = protocol.Message()
                msg.object_state_update.CopyFrom(protocol.ObjectStateUpdate())
                msg.object_state_update.object_id = uid
                # msg.object_state_update.field.field_name = field_name
                fu = protocol.ObjectStateUpdate.FieldUpdate()
                fu.field_name = field_name
                if type(value[0]) is int:
                    fu.int_value.extend(value)
                if type(value[0]) is float:
                    fu.float_value.extend(value)
                if type(value[0]) is str:
                    fu.string_value.extend(value)
                msg.object_state_update.field.extend([fu])
                self.server_socket.sendall(msg.SerializeToString())

            if len(self.creation_queue)>0:
                print("Object being created")
                asker = self.creation_queue.pop()
                self.creation_queue_waiting_answer.append(asker)
                msg = protocol.Message()
                msg.create_obj.CopyFrom(protocol.CreateObj())
                self.server_socket.sendall(msg.SerializeToString())
                print("Send:", msg)

    def queue_update_field(self, uid, field_name, value):
        if str(value) is str or not hasattr(value, "__getitem__"):
            value = [value]
        self.updates_queue.append((uid, field_name, value))

    def queue_object_creation(self, asker):
        print("Queue object creation")
        self.creation_queue.append(asker)
