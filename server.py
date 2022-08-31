"""
TODO

- "Auth" players ✓
- UserActions:
    Click object
- Scene updates:
    Object creation
    Object destruction
    Object field update
"""

import socketserver
import threading
from time import sleep

import wol.protobuf.message_pb2 as protocol


class SyncedObj:
    def __init__(self, uid):
        self.fields = dict()
        self.updated_fields = list()
        self.uid = uid


class DataStore:
    def __init__(self):
        self.next_uid = 1
        self.users = dict()                 # name to UID
        self.objs_by_id = dict()
        self.connections = dict()           # IP to socket
        self.connections_uid = dict()       # IP to UID
        self.updated_objs = list()
        self.lock = threading.Lock()

    def new_uid(self):
        a = self.next_uid
        self.next_uid += 1
        return a

    def add_user(self, name, address, socket):
        self.lock.acquire()
        if name not in self.users.keys():
            uid = self.users[name] = self.new_uid()
        else:
            uid = self.users[name]

        self.connections[address] = socket
        self.connections_uid[address] = uid
        self.lock.release()
        return uid

    def add_node(self):
        self.lock.acquire()
        uid = self.new_uid()
        self.objs_by_id[uid]=SyncedObj(uid)
        self.lock.release()
        return uid

    def update_field(self, uid, field, value):
        self.lock.acquire()
        o = self.objs_by_id[uid]
        o.fields[field] = value
        o.updated_fields.append((field, value))
        self.updated_objs.append(o)
        self.lock.release()


class ThreadedTCPRequestHandler(socketserver.StreamRequestHandler):
    # Handles one client
    def handle(self):
        received = self.request.recv(1024)
        print(received)
        msg = protocol.Message()
        msg.ParseFromString(received)

        if msg.WhichOneof("content") == "auth":
            uid = self.server.ds.add_user(msg.auth.name, self.client_address, self.request)
            response = protocol.Message()
            response.auth_response.CopyFrom(protocol.AuthResponse())
            response.auth_response.success = True
            response.auth_response.your_id = uid
            self.request.sendall(response.SerializeToString())
            self.server.alive_users.add(self.request)

        while self.request in self.server.alive_users:
            received = self.request.recv(1024)
            if len(received) == 0:
                continue
            print(received)
            msg = protocol.Message()
            msg.ParseFromString(received)
            print(f"Received one request from {self.client_address}: {msg}")

            response = None
            if msg.WhichOneof("content") == "auth":
                uid = self.server.ds.add_user(msg.auth.name, self.client_address, self.request)
                response = protocol.Message()
                response.auth_response.CopyFrom(protocol.AuthResponse())
                response.auth_response.success = True
                response.auth_response.your_id = uid

            if msg.WhichOneof("content") == "create_obj":
                uid = self.server.ds.add_node()
                response = protocol.Message()
                response.new_object_create.CopyFrom(protocol.NewObjectCreated())
                response.new_object_create.object_id = uid;

            if msg.WhichOneof("content") == "object_state_update":
                for f in msg.object_state_update.field:
                    self.server.ds.update_field(msg.object_state_update.object_id,
                                                f.field_name,
                                                list(f.float_value))

            if response:
                print(response)
                self.request.sendall(response.SerializeToString())


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    def __init__(self, *args):
        super(ThreadedTCPServer, self).__init__(*args)
        self.ds = DataStore()
        self.alive_users = set()            # sockets
        self.send_thread = threading.Thread(target=self.send_loop)
        self.send_thread.start()

    def send_loop(self):
        while True:
            sleep(0.01)
            while len(self.ds.updated_objs) > 0:
                self.ds.lock.acquire()
                obj = self.ds.updated_objs.pop()
                print(f"sending updates for uid {obj.uid}")
                for name, value in obj.updated_fields:
                    msg = protocol.Message()
                    msg.object_state_update.CopyFrom(protocol.ObjectStateUpdate())
                    msg.object_state_update.object_id = obj.uid
                    fu = protocol.ObjectStateUpdate.FieldUpdate()
                    fu.field_name = name
                    if str(value) is str or not hasattr(value, "__getitem__"):
                        value = [value]
                    if type(value[0]) is int:
                        fu.int_value.extend(value)
                    if type(value[0]) is float:
                        fu.float_value.extend(value)
                    if type(value[0]) is str:
                        fu.string_value.extend(value)
                    msg.object_state_update.field.extend([fu])
                    for socket in self.alive_users:
                        socket.sendall(msg.SerializeToString())
                self.ds.lock.release()


port = 8971
host = 'localhost'
socketserver.TCPServer.allow_reuse_address = True
server = ThreadedTCPServer((host, port), ThreadedTCPRequestHandler)
server.serve_forever()
