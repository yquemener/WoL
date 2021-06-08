#!/usr/bin/env python
import re
import sys
import signal
import time

import pybullet
import pybullet as pb

from PyQt5 import QtCore
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtWidgets import QApplication, QTextEdit, QMainWindow, QVBoxLayout, QHBoxLayout, QDialog
from PyQt5.QtGui import QVector3D, QQuaternion, QOpenGLTexture, QImage

from wol import Behavior, utils
from wol.Behavior import Focusable

from wol.Constants import UserActions, Events
from wol.Notebook import NotebookNode
from wol.SceneNode import CameraNode, SkyBox
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
    context.scene.context.current_camera = my_cam

    SkyBox(parent=context.current_camera)

    g = Grid(parent=context.scene)
    g.orientation = QQuaternion.fromEulerAngles(0.0, 0.0, 90.0)

    window.load_scene()

    create_button = CubeNode(parent=context.scene, name="CreateNotebookButton")
    create_button.tooltip = "Create New Notebook"
    create_button.events_handlers[Events.Clicked].append(lambda: create_new_notebook(context))
    create_button.scale = QVector3D(0.2, 0.2, 0.2)
    create_button.position = QVector3D(-3, 5, 0.2)
    create_button.properties["skip serialization"] = True

    sph = Sphere(name="SpherePointer", parent=context.scene)
    sph.scale = QVector3D(0.2, 0.2, 0.2)
    pb.removeBody(sph.collider_id)
    sph.collider_id = None
    sph.visible = False
    context.debug_sphere = sph

    mesh = MeshNode(filename="urdf/duck.obj", parent=context.scene)
    sph = Sphere(parent=context.scene)
    sph.position = QVector3D(1,0,0)

    # card = CardNode(name="CardTest", parent=context.scene, filename="test.png")
    # card.position = QVector3D(5, 3, 0)


    # nb = NotebookNode(parent=context.scene, name="Notebook2")
    # nb.position = QVector3D(-1, 5, 0)
    # nb.add_cell(1)
    # nb.cells[0].set_text("print('Hi!')")

    # db = Sphere(parent=context.scene)
    # db.scale = QVector3D(0.01,0.01,0.01)
    # db.add_behavior(DebugBehavior())

    # window2 = QDialog()
    # layout = QHBoxLayout()
    # layout.addWidget(window)
    # layout.addWidget(QTextEdit())
    # window2.setLayout(layout)
    # window2.show()


    window.show()
    window.setFocus()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(app.exec_())

