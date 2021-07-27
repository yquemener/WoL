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
cell.set_text('rm(9)')
cell = obj_2.add_cell(5)
cell.set_text('')
cell = obj_2.layout()

obj_3 = wol.GeomNodes.Sphere(parent=context.scene,name="SpherePointer")
obj_3.position = PyQt5.QtGui.QVector3D(0.0, 0.0, 0.0)
obj_3.orientation = PyQt5.QtGui.QQuaternion(1.0, 0.0, 0.0, 0.0)
obj_3.scale = PyQt5.QtGui.QVector3D(0.20000000298023224, 0.20000000298023224, 0.20000000298023224)
obj_3.properties = {}
obj_3.color = PyQt5.QtGui.QVector4D(0.5, 1.0, 0.5, 1.0)
obj_3.visible = False

obj_4 = wol.Notebook.NotebookNode(parent=context.scene,name="Notebook8")
obj_4.position = PyQt5.QtGui.QVector3D(4.493555068969727, 4.248137474060059, 3.0554356575012207)
obj_4.orientation = PyQt5.QtGui.QQuaternion(0.9537873268127441, -0.061887599527835846, -0.29342329502105713, -0.019039127975702286)
obj_4.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_4.properties = {}
obj_4.visible = True
cell = obj_4.add_cell(0)
cell.set_text('import pybullet as pb\nimport time')
cell = obj_4.add_cell(1)
cell.set_text('print(plane)')
cell = obj_4.add_cell(2)
cell.set_text('a=pb.loadURDF("urdf/Pendulum_Tendon_1_Cart_Rail.urdf")')
cell = obj_4.add_cell(3)
cell.set_text('print(a)')
cell = obj_4.add_cell(4)
cell.set_text('nJoints = pb.getNumJoints(a)\nfor i in range(nJoints):\n  info = pb.getJointInfo(a, i)\n  print(pb.getBasePositionAndOrientation(info[0]))')
cell = obj_4.add_cell(5)
cell.set_text('from wol.SceneNode import SceneNode\nfrom wol.GeomNodes import Sphere\nfrom PyQt5.QtGui import QVector3D')
cell = obj_4.add_cell(6)
cell.set_text('nJoints = pb.getNumJoints(a)\nfor i in range(nJoints):\n  info = pb.getJointInfo(a, i)\n  pos,orient = pb.getBasePositionAndOrientation(info[0])\n  o =  Sphere(parent=scene)\n  o.position = QVector3D(pos[0], pos[1], pos[2])')
cell = obj_4.add_cell(7)
cell.set_text('')
cell = obj_4.add_cell(8)
cell.set_text('pb.setGravity(0,0,-10)\nfor i in range(100):\n\tpb.stepSimulation()\n\ttime.sleep(0.02)')
cell = obj_4.add_cell(9)
cell.set_text('')
cell = obj_4.layout()

obj_5 = wol.SceneNode.SceneNode(parent=context.scene,name="Node")
obj_5.position = PyQt5.QtGui.QVector3D(0.0, 0.0, 0.0)
obj_5.orientation = PyQt5.QtGui.QQuaternion(1.0, 0.0, 0.0, 0.0)
obj_5.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_5.properties = {}
obj_5.visible = True

obj_6 = wol.SceneNode.SceneNode(parent=context.scene,name="Node")
obj_6.position = PyQt5.QtGui.QVector3D(0.0, 0.0, 0.0)
obj_6.orientation = PyQt5.QtGui.QQuaternion(1.0, 0.0, 0.0, 0.0)
obj_6.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_6.properties = {}
obj_6.visible = True

obj_7 = wol.Notebook.NotebookNode(parent=context.scene,name="Notebook18")
obj_7.position = PyQt5.QtGui.QVector3D(-2.831247329711914, 3.3779945373535156, 3.2415881156921387)
obj_7.orientation = PyQt5.QtGui.QQuaternion(0.9977447390556335, -0.03854750096797943, 0.05490931496024132, 0.0021214094012975693)
obj_7.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_7.properties = {}
obj_7.visible = True
cell = obj_7.add_cell(0)
cell.set_text('import pybullet as pb\nimport time\n')
cell = obj_7.add_cell(1)
cell.set_text('pb.setGravity(0,0,-10)\nfor i in range(1000):\n\tpb.stepSimulation()\n\ttime.sleep(1./240.)')
cell = obj_7.add_cell(2)
cell.set_text('')
cell = obj_7.layout()

obj_8 = wol.Notebook.NotebookNode(parent=context.scene,name="Notebook14")
obj_8.position = PyQt5.QtGui.QVector3D(2.826246500015259, 4.842960834503174, 7.1737542152404785)
obj_8.orientation = PyQt5.QtGui.QQuaternion(0.9237014055252075, -0.01813916303217411, -0.3826097249984741, -0.0075134942308068275)
obj_8.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_8.properties = {}
obj_8.visible = True
cell = obj_8.add_cell(0)
cell.set_text('import odepy\nimport ctypes')
cell = obj_8.add_cell(1)
cell.set_text("def AddBall(world, space, m=1.0, r=0.1, pos=[0, 0, 2.0]):\n    mass = odepy.dMass()\n    odepy.dMassSetZero(ctypes.byref(mass))\n    odepy.dMassSetSphereTotal(ctypes.byref(mass), m, r)\n    body = odepy.dBodyCreate(world)\n    odepy.dBodySetMass(body, ctypes.byref(mass))\n    geom = odepy.dCreateSphere(space, r)\n    odepy.dGeomSetBody(geom, body)\n    ball = {'body': body, 'geom': geom}\n    odepy.dBodySetPosition(ball['body'], *pos)\n    return ball")
cell = obj_8.add_cell(2)
cell.set_text('def CreateWorld():\n    odepy.dInitODE()\n    world = odepy.dWorldCreate()\n    odepy.dWorldSetGravity(world, 0, 0, -9.8)\n    space = odepy.dHashSpaceCreate(0)\n    ground = odepy.dCreatePlane(space, 0, 0, 1, 0)\n    contactgroup = odepy.dJointGroupCreate(0)\n    return world, space, ground, contactgroup\n    \ndef NearCallback(*args):\n\treturn')
cell = obj_8.add_cell(3)
cell.set_text("world, space, ground, contactgroup = CreateWorld()\ngeoms = [AddBall(world, space)['geom']]")
cell = obj_8.add_cell(4)
cell.set_text('for k in range(100):\n\todepy.dSpaceCollide(space, 0, odepy.dNearCallback(NearCallback))\n\todepy.dWorldStep(world, 0.1)\n\todepy.dJointGroupEmpty(contactgroup)\n\tfor geom in geoms:\n\t\tbody = odepy.dGeomGetBody(geom)\n\t\tif odepy.dGeomGetClass(geom) == odepy.dSphereClass:\n\t\t\tpos = odepy.dBodyGetPosition(body)\n\t\t\tprint(pos[0], pos[1], pos[2])\n       \t')
cell = obj_8.add_cell(5)
cell.set_text('')
cell = obj_8.layout()

obj_9 = wol.Notebook.NotebookNode(parent=context.scene,name="Notebook12")
obj_9.position = PyQt5.QtGui.QVector3D(1.9375934600830078, 5.505330562591553, 11.963836669921875)
obj_9.orientation = PyQt5.QtGui.QQuaternion(0.7498143315315247, -0.021108144894242287, -0.6610496044158936, -0.018609285354614258)
obj_9.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_9.properties = {}
obj_9.visible = True
cell = obj_9.add_cell(0)
cell.set_text("print('Hi!')")
cell = obj_9.add_cell(1)
cell.set_text('')
cell = obj_9.layout()

obj_10 = wol.Notebook.NotebookNode(parent=context.scene,name="Notebook14")
obj_10.position = PyQt5.QtGui.QVector3D(-6.501605033874512, 3.4106671810150146, 2.3507437705993652)
obj_10.orientation = PyQt5.QtGui.QQuaternion(0.9322095513343811, -0.07772929966449738, 0.35209545493125916, 0.03118155710399151)
obj_10.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_10.properties = {}
obj_10.visible = True
cell = obj_10.add_cell(0)
cell.set_text('import pybullet as p\nfrom PyQt5.QtGui import QVector3D, QQuaternion, QOpenGLTexture, QImage\nfrom wol.GeomNodes import Grid, Sphere, CubeNode, CardNode, MeshNode, UrdfNode, UrdfSingleNode\n\n#plane = p.loadURDF("plane.urdf", [0,0,0], p.getQuaternionFromEuler([0.1, 0, 0]))')
cell = obj_10.add_cell(1)
cell.set_text('cubeStartPos3 = [0, 0, 20]\n\nPulleyStartOrientation = p.getQuaternionFromEuler([1.570796, 0, 0])\ncubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])\ncubeStartOrientation2 = p.getQuaternionFromEuler([0, -1.570796, 0])\n\n#ball = p.loadURDF("sphere.urdf", cubeStartPos3, cubeStartOrientation)\n#ball = UrdfNode("sphere.urdf", parent=scene)\nball=UrdfSingleNode("sphere.urdf", parent=scene)\nball.position = QVector3D(0,0,20)\np.setGravity(0, 0, -10)')
cell = obj_10.add_cell(2)
cell.set_text('import time\n\nfor i in range(100):\n    p.stepSimulation()\n    print(ball.position)\n    #print(p.getBasePositionAndOrientation(ball))\n    time.sleep(1. / 24.)')
cell = obj_10.add_cell(3)
cell.set_text('print(ball.position)')
cell = obj_10.add_cell(4)
cell.set_text('')
cell = obj_10.layout()

obj_11 = wol.Notebook.NotebookNode(parent=context.scene,name="Notebook13")
obj_11.position = PyQt5.QtGui.QVector3D(-0.33007359504699707, 2.176287889480591, 5.102674961090088)
obj_11.orientation = PyQt5.QtGui.QQuaternion(0.9339178800582886, -0.20168274641036987, -0.2855578362941742, -0.07468751817941666)
obj_11.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_11.properties = {}
obj_11.visible = True
cell = obj_11.add_cell(0)
cell.set_text('from wol.GeomNodes import *\nfrom PyQt5.QtGui import QVector3D, QQuaternion, QOpenGLTexture, QImage')
cell = obj_11.add_cell(1)
cell.set_text('s=Sphere(parent=scene)')
cell = obj_11.add_cell(2)
cell.set_text('s.add_behavior(OdeSphereBehavior(0.5,obj=s))\nodepy.dWorldSetGravity(scene.context.ode_world, 0, 0, 0)')
cell = obj_11.add_cell(3)
cell.set_text('import time\n\ndef NearCallback(*args):\n\tprint(args)\n\nfor k in range(1000):\n\todepy.dWorldStep(scene.context.ode_world, 0.1)\n\todepy.dSpaceCollide(scene.context.ode_space, 0, odepy.dNearCallback(NearCallback))\n\todepy.dJointGroupEmpty(scene.context.ode_contactgroup)\n\ttime.sleep(0.1)\n\t\n\n')
cell = obj_11.add_cell(4)
cell.set_text('')
cell = obj_11.add_cell(5)
cell.set_text('')
cell = obj_11.layout()




