#!/usr/bin/env python
import re
import sys
import signal
import time

import PyQt5
from PyQt5 import QtCore
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtWidgets import QApplication, QTextEdit, QMainWindow, QVBoxLayout, QHBoxLayout, QDialog
from PyQt5.QtGui import QVector3D, QQuaternion, QOpenGLTexture, QImage

import wol
from wol import Behavior, utils, GeomNodes
from wol.Behavior import Focusable, RotateConstantSpeed

from wol.Constants import UserActions, Events
from wol.Notebook import NotebookNode
from wol.SceneNode import CameraNode, SkyBox, SceneNode
from wol.GeomNodes import Grid, Sphere, CubeNode, CardNode, MeshNode
from wol.TextEditNode import TextEditNode
from wol.View3D import View3D


class HUDEditor(TextEditNode):
    def __init__(self, parent, name):
        TextEditNode.__init__(self, parent=parent, name=name)
        self.target_object = None
        self.layer = 0
        self.events_handlers[UserActions.Unselect].append(self.unfocus)

    def keyPressEvent(self, evt):
        if evt.key() == QtCore.Qt.Key_Return and \
                QApplication.keyboardModifiers() == QtCore.Qt.ShiftModifier:
            self.target_object.set_text(self.text)
            self.target_object.on_event(UserActions.Execute)
            self.unfocus()
        else:
            super().keyPressEvent(evt)
            self.needs_refresh = True
            self.target_object.keyPressEvent(evt)
            self.target_object.needs_refresh = True

    def unfocus(self):
        self.visible = False


class DebugBehavior(Behavior.Behavior):
    def on_update(self, dt):
        self.obj.position = self.obj.context.debug_point


class ShowOrient(Behavior.Behavior):
    def on_update(self, dt):
        print(self.obj.orientation)
        print(self.obj.world_orientation())
        print(self.obj.parent.orientation)
        print(self.obj.parent.world_orientation())
        print()


def create_new_notebook(ctxt):
    nb = NotebookNode(parent=ctxt.scene, name="Notebook"+str(len(ctxt.scene.children)))
    stc = ctxt.current_camera.get_behavior("SnapToCamera")
    ctxt.hover_target = nb
    stc.on_grab()
    nb.position = QVector3D(0, 0, 4)
    nb.orientation = QQuaternion.fromAxisAndAngle(0, 1, 0, 180)
    nb.add_cell(1)
    nb.cells[0].set_text("print('Hi!')")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = View3D()
    context = window.context

    my_cam = CameraNode(parent=context.scene, name="MainCamera")
    my_cam.speed = 0.2
    my_cam.position = QVector3D(0, 5, 10)
    my_cam.add_behavior(Behavior.MoveAround(0.2))
    my_cam.add_behavior(Behavior.SnapToCamera())
    my_cam.ray = GeomNodes.OdeRayBehavior(obj=my_cam)
    my_cam.add_behavior(my_cam.ray)

    context.scene.context.current_camera = my_cam

    SkyBox(parent=context.current_camera)

    g = Grid(parent=context.scene)
    g.orientation = QQuaternion.fromEulerAngles(0.0, 0.0, 90.0)

    window.load_scene()

    # obj_4 = wol.Notebook.NotebookNode(parent=context.scene, name="Notebook8")
    # obj_4.position = PyQt5.QtGui.QVector3D(4.493555068969727, 4.248137474060059, 3.0554356575012207)
    # obj_4.orientation = PyQt5.QtGui.QQuaternion(0.9537873268127441, -0.061887599527835846, -0.29342329502105713,
    #                                             -0.019039127975702286)
    # obj_4.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
    # obj_4.properties = {}
    # obj_4.visible = True
    # cell = obj_4.add_cell(0)
    # cell.set_text('import pybullet as pb\nimport time')
    #
    # obj_4.add_behavior(RotateConstantSpeed())
    # # obj_4.cells[0].add_behavior(ShowOrient())
    # print("Scale = ", obj_4.scale)

    create_button = CubeNode(parent=context.scene, name="CreateNotebookButton")
    create_button.tooltip = "Create New Notebook"
    create_button.events_handlers[Events.Clicked].append(lambda: create_new_notebook(context))
    create_button.scale = QVector3D(0.7, 0.2, 0.2)
    create_button.position = QVector3D(-3, 5, 0.2)
    create_button.on_event(Events.GeometryChanged)
    create_button.properties["skip serialization"] = True


    # p1 = SceneNode(parent=context.scene, name="p1")
    # p1.orientation = PyQt5.QtGui.QQuaternion(0.9537873268127441, -0.061887599527835846, -0.29342329502105713,
    #                                             -0.019039127975702286)
    # p2 = SceneNode(parent=p1, name="p2")
    # p2.scale = QVector3D(0.5,0.5,0.5)
    # p1.add_behavior(RotateConstantSpeed())
    # # p1.add_behavior(ShowOrient())
    # p2.add_behavior(ShowOrient())

    sph = Sphere(name="SpherePointer", parent=context.scene)
    sph.scale = QVector3D(0.05, 0.05, 0.05)
    # sph.scale = QVector3D(0.2, 0.2, 0.2)
    sph.collider_id = None
    sph.visible = False
    context.debug_sphere = sph
    sph.properties["skip serialization"] = True

    window.show()
    window.setFocus()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(app.exec_())

