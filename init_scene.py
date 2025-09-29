import wol
import PyQt5


obj_0 = wol.GeomNodes.Grid(parent=context.scene,name="Grid")
obj_0.position = PyQt5.QtGui.QVector3D(0.0, 0.0, 0.0)
obj_0.orientation = PyQt5.QtGui.QQuaternion(0.7071067690849304, 0.0, 0.0, 0.7071067690849304)
obj_0.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_0.properties = {}
obj_0.color = PyQt5.QtGui.QVector4D(0.20000000298023224, 0.20000000298023224, 0.6000000238418579, 1.0)
obj_0.visible = True

obj_1 = wol.GeomNodes.Avatar(parent=context.scene,name="avatar_tata")
obj_1.position = PyQt5.QtGui.QVector3D(0.0, 0.0, 0.0)
obj_1.orientation = PyQt5.QtGui.QQuaternion(0.9928369522094727, 0.030225396156311035, -0.11553698778152466, 0.0035172980278730392)
obj_1.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_1.properties = {}
obj_1.visible = True

obj_2 = wol.Notebook.NotebookNode(parent=context.scene,name="Notebook5")
obj_2.position = PyQt5.QtGui.QVector3D(-1.4135291576385498, 5.406454563140869, 1.3666977882385254)
obj_2.orientation = PyQt5.QtGui.QQuaternion(0.9890741109848022, -0.0030229396652430296, -0.14738093316555023, -0.0014766992535442114)
obj_2.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_2.properties = {}
obj_2.visible = True
cell = obj_2.add_cell(0)
cell.set_text("print('Hi!')")
cell = obj_2.add_cell(1)
cell.set_text('a=12')
cell = obj_2.add_cell(2)
cell.set_text('print(a)')
cell = obj_2.add_cell(3)
cell.set_text('')
cell = obj_2.layout()

obj_3 = wol.Notebook.NotebookNode(parent=context.scene,name="Notebook7")
obj_3.position = PyQt5.QtGui.QVector3D(-1.3406577110290527, 5.601081848144531, 10.698970794677734)
obj_3.orientation = PyQt5.QtGui.QQuaternion(0.8692184686660767, -0.07008276879787445, -0.48785293102264404, -0.039334263652563095)
obj_3.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_3.properties = {}
obj_3.visible = True
cell = obj_3.add_cell(0)
cell.set_text("print('Hi!')")
cell = obj_3.add_cell(1)
cell.set_text('')
cell = obj_3.layout()

obj_4 = wol.Notebook.NotebookNode(parent=context.scene,name="Notebook8")
obj_4.position = PyQt5.QtGui.QVector3D(0.17542731761932373, 5.30002498626709, 2.951037645339966)
obj_4.orientation = PyQt5.QtGui.QQuaternion(0.9768542647361755, -0.03740271180868149, -0.2104564756155014, -0.008058151230216026)
obj_4.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_4.properties = {}
obj_4.visible = True
cell = obj_4.add_cell(0)
cell.set_text("print('Hi!')")
cell = obj_4.add_cell(1)
cell.set_text('')
cell = obj_4.layout()



obj_7 = wol.Notebook.NotebookNode(parent=context.scene,name="Notebook8")
obj_7.position = PyQt5.QtGui.QVector3D(-2.1454219818115234, 5.360567092895508, 5.775728702545166)
obj_7.orientation = PyQt5.QtGui.QQuaternion(0.9050676822662354, 0.039713941514492035, -0.42300209403038025, 0.01856112666428089)
obj_7.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_7.properties = {}
obj_7.visible = True
cell = obj_7.add_cell(0)
cell.set_text("print('Hi!')")
cell = obj_7.add_cell(1)
cell.set_text('print(context)')
cell = obj_7.add_cell(2)
cell.set_text('print(scene)')
cell = obj_7.add_cell(3)
cell.set_text('for c in scene.children:\n\tprint(c.name, type(c))')
cell = obj_7.add_cell(4)
cell.set_text('')
cell = obj_7.layout()
