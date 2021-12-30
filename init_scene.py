import wol
import PyQt5


obj_0 = wol.GeomNodes.Grid(parent=context.scene,name="Grid")
obj_0.position = PyQt5.QtGui.QVector3D(0.0, 0.0, 0.0)
obj_0.orientation = PyQt5.QtGui.QQuaternion(0.7071067690849304, 0.0, 0.0, 0.7071067690849304)
obj_0.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_0.properties = {}
obj_0.color = PyQt5.QtGui.QVector4D(0.20000000298023224, 0.20000000298023224, 0.6000000238418579, 1.0)
obj_0.visible = True

obj_1 = wol.Notebook.NotebookNode(parent=context.scene,name="Notebook")
obj_1.position = PyQt5.QtGui.QVector3D(0.0, 5.0, 0.0)
obj_1.orientation = PyQt5.QtGui.QQuaternion(1.0, 0.0, 0.0, 0.0)
obj_1.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_1.properties = {}
obj_1.visible = True
cell = obj_1.add_cell(0)
cell.set_text('import torch\nimport torch.nn as nn\nimport torch.optim as optim\nimport numpy as np\n')
cell = obj_1.add_cell(1)
cell.set_text('class GRUBasic(nn.Module):\n    def __init__(self, in_size, hidden_size):\n        super(GRUBasic, self).__init__()\n        self.gru = nn.GRU(in_size, hidden_size, batch_first=True)\n        self.dense1 = nn.Linear(hidden_size, 10)\n        self.dense2 = nn.Linear(10, 1)\n\n    def forward(self, sequence):\n        gru_out, _ = self.gru(sequence)\n        x = self.dense1(gru_out[:,-1,:])\n        x = torch.relu(x)\n        x = self.dense2(x)\n        return x')
cell = obj_1.add_cell(2)
cell.set_text('g=GRUBasic(3,5)')
cell = obj_1.add_cell(3)
cell.set_text("watch('torch.Tensor(g.dense1.weight).detach().numpy()*255')")
cell = obj_1.add_cell(4)
cell.set_text('print(g.dense1.out_features)')
cell = obj_1.add_cell(5)
cell.set_text('for a in dir(g.dense1):\n\tif not a.startswith("__"):\n\t\tprint(a)')
cell = obj_1.add_cell(6)
cell.set_text('import time\n\nfor i in range(50):\n\tins = torch.Tensor(np.random.random((3,3,3)))\n\ttime.sleep(1)\n')
cell = obj_1.add_cell(7)
cell.set_text("watch('torch.Tensor(g(ins)).detach().numpy()*255')")
cell = obj_1.add_cell(8)
cell.set_text("watch('torch.Tensor(ins).detach().numpy()*255')")
cell = obj_1.add_cell(9)
cell.set_text('for i in range(10):\n\tprint(i)')
cell = obj_1.add_cell(10)
cell.set_text('')
cell = obj_1.layout()

obj_2 = wol.Notebook.NotebookNode(parent=context.scene,name="Notebook6")
obj_2.position = PyQt5.QtGui.QVector3D(2.71406888961792, 3.9321537017822266, 0.7638711929321289)
obj_2.orientation = PyQt5.QtGui.QQuaternion(1.0, -1.4901162970204496e-08, -1.4901162970204496e-08, 3.725290298461914e-09)
obj_2.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_2.properties = {}
obj_2.visible = True
cell = obj_2.add_cell(0)
cell.set_text('from wol.SceneNode import SceneNode\nfrom wol.GeomNodes import Sphere\nfrom wol.Behavior import Behavior\nfrom math import *\nimport time\nfrom PyQt5.QtGui import QVector3D')
cell = obj_2.add_cell(1)
cell.set_text('g=Sphere(parent=scene)')
cell = obj_2.add_cell(2)
cell.set_text('class move(Behavior):\n\tdef on_update(self, dt):\n\t\tself.obj.position=QVector3D(cos(time.time()), 0, 0)')
cell = obj_2.add_cell(3)
cell.set_text('g.add_behavior(move())')
cell = obj_2.add_cell(4)
cell.set_text('ls()')
cell = obj_2.add_cell(5)
cell.set_text('rm(9)')
cell = obj_2.add_cell(6)
cell.set_text('')
cell = obj_2.layout()

obj_3 = wol.GeomNodes.Sphere(parent=context.scene,name="SpherePointer")
obj_3.position = PyQt5.QtGui.QVector3D(0.0, 0.0, 0.0)
obj_3.orientation = PyQt5.QtGui.QQuaternion(1.0, 0.0, 0.0, 0.0)
obj_3.scale = PyQt5.QtGui.QVector3D(0.20000000298023224, 0.20000000298023224, 0.20000000298023224)
obj_3.properties = {}
obj_3.color = PyQt5.QtGui.QVector4D(0.5, 1.0, 0.5, 1.0)
obj_3.visible = False

obj_4 = wol.SceneNode.SceneNode(parent=context.scene,name="Node")
obj_4.position = PyQt5.QtGui.QVector3D(0.0, 0.0, 0.0)
obj_4.orientation = PyQt5.QtGui.QQuaternion(1.0, 0.0, 0.0, 0.0)
obj_4.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_4.properties = {}
obj_4.visible = True

obj_5 = wol.SceneNode.SceneNode(parent=context.scene,name="Node")
obj_5.position = PyQt5.QtGui.QVector3D(0.0, 0.0, 0.0)
obj_5.orientation = PyQt5.QtGui.QQuaternion(1.0, 0.0, 0.0, 0.0)
obj_5.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_5.properties = {}
obj_5.visible = True

obj_6 = wol.Notebook.NotebookNode(parent=context.scene,name="Notebook10")
obj_6.position = PyQt5.QtGui.QVector3D(-4.091843605041504, 4.062075138092041, 3.209451198577881)
obj_6.orientation = PyQt5.QtGui.QQuaternion(0.9938439130783081, -0.05404214933514595, 0.09657181054353714, 0.005251279100775719)
obj_6.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_6.properties = {}
obj_6.visible = True
cell = obj_6.add_cell(0)
cell.set_text('import socket\n\ns = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\ns.sendto(bytes("Hi! Yves", \'ascii\'), (\'127.0.0.1\', 8971))')
cell = obj_6.add_cell(1)
cell.set_text('s.sendto(bytes("pos 1 1 1", \'ascii\'), (\'127.0.0.1\', 8971))')
cell = obj_6.add_cell(2)
cell.set_text('')
cell = obj_6.layout()

obj_7 = wol.Notebook.NotebookNode(parent=context.scene,name="Notebook14")
obj_7.position = PyQt5.QtGui.QVector3D(2.7049832344055176, 5.102869510650635, 7.049956798553467)
obj_7.orientation = PyQt5.QtGui.QQuaternion(0.928321897983551, -0.0018227876862511039, -0.37177222967147827, -0.0007299954886548221)
obj_7.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_7.properties = {}
obj_7.visible = True
cell = obj_7.add_cell(0)
cell.set_text('import odepy\nimport ctypes')
cell = obj_7.add_cell(1)
cell.set_text("def AddBall(world, space, m=1.0, r=0.1, pos=[0, 0, 2.0]):\n    mass = odepy.dMass()\n    odepy.dMassSetZero(ctypes.byref(mass))\n    odepy.dMassSetSphereTotal(ctypes.byref(mass), m, r)\n    body = odepy.dBodyCreate(world)\n    odepy.dBodySetMass(body, ctypes.byref(mass))\n    geom = odepy.dCreateSphere(space, r)\n    odepy.dGeomSetBody(geom, body)\n    ball = {'body': body, 'geom': geom}\n    odepy.dBodySetPosition(ball['body'], *pos)\n    return ball")
cell = obj_7.add_cell(2)
cell.set_text('def CreateWorld():\n    odepy.dInitODE()\n    world = odepy.dWorldCreate()\n    odepy.dWorldSetGravity(world, 0, 0, -9.8)\n    space = odepy.dHashSpaceCreate(0)\n    ground = odepy.dCreatePlane(space, 0, 0, 1, 0)\n    contactgroup = odepy.dJointGroupCreate(0)\n    return world, space, ground, contactgroup\n    \ndef NearCallback(*args):\n\treturn')
cell = obj_7.add_cell(3)
cell.set_text("world, space, ground, contactgroup = CreateWorld()\ngeoms = [AddBall(world, space)['geom']]")
cell = obj_7.add_cell(4)
cell.set_text('for k in range(100):\n\todepy.dSpaceCollide(space, 0, odepy.dNearCallback(NearCallback))\n\todepy.dWorldStep(world, 0.1)\n\todepy.dJointGroupEmpty(contactgroup)\n\tfor geom in geoms:\n\t\tbody = odepy.dGeomGetBody(geom)\n\t\tif odepy.dGeomGetClass(geom) == odepy.dSphereClass:\n\t\t\tpos = odepy.dBodyGetPosition(body)\n\t\t\tprint(pos[0], pos[1], pos[2])\n       \t')
cell = obj_7.add_cell(5)
cell.set_text('')
cell = obj_7.layout()

obj_8 = wol.Notebook.NotebookNode(parent=context.scene,name="Notebook11")
obj_8.position = PyQt5.QtGui.QVector3D(-5.650867462158203, 4.793712615966797, 4.563736915588379)
obj_8.orientation = PyQt5.QtGui.QQuaternion(0.9119216203689575, 0.005968572571873665, 0.4103122055530548, -0.0026854875031858683)
obj_8.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_8.properties = {}
obj_8.visible = True
cell = obj_8.add_cell(0)
cell.set_text('o=Sphere(name="n", parent=scene)')
cell = obj_8.add_cell(1)
cell.set_text('ls()')
cell = obj_8.add_cell(2)
cell.set_text('rm(10)')
cell = obj_8.add_cell(3)
cell.set_text('')
cell = obj_8.layout()



obj_11 = wol.GeomNodes.Sphere(parent=context.scene,name="avatar_Yves2")
obj_11.position = PyQt5.QtGui.QVector3D(-27.636117935180664, 7.1307501792907715, 10.11546802520752)
obj_11.orientation = PyQt5.QtGui.QQuaternion(1.0, 0.0, 0.0, 0.0)
obj_11.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_11.properties = {}
obj_11.color = PyQt5.QtGui.QVector4D(0.5, 1.0, 0.5, 1.0)
obj_11.visible = True
