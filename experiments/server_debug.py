import socketserver
import threading
import message_pb2 as protocol


class ThreadedTCPRequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        while True:
            received = self.request.recv(1024)
            if len(received)==0:
                continue
            print(received)
            msg = protocol.Message()
            msg.ParseFromString(received)
            print(f"Received one request from {self.client_address}: {msg}")

            response = protocol.Message()
            if msg.WhichOneof("content") == "auth":
                # uid = self.server.ds.add_user(msg.auth.name, self.client_address, self.request)
                response.auth_response.CopyFrom(protocol.AuthResponse())
                response.auth_response.success = True
                response.auth_response.your_id = 5

            if msg.WhichOneof("content") == "CreateObj":
                # uid = self.server.ds.add_node()
                response.new_object_create.CopyFrom(protocol.NewObjectCreated())
                response.new_object_create.object_id = 15

            print(response)
            # self.request.sendall(response.SerializeToString())
        print("Exit handle")


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    def __init__(self, *args):
        super(ThreadedTCPServer, self).__init__(*args)


port = 8971
host = 'localhost'
socketserver.TCPServer.allow_reuse_address = True
server = ThreadedTCPServer((host, port), ThreadedTCPRequestHandler)
server.serve_forever()
