import message_pb2

# auth = message_pb2.Auth()
# auth.name = "Yves"
#
m = message_pb2.Message()
# m.auth.CopyFrom(auth)

m.auth.CopyFrom(message_pb2.Auth())
m.auth.name = "Yves"

s = m.SerializeToString()
print(s)

m2 = message_pb2.Message()
m2.ParseFromString(s)
print(m2)