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
from wol.NetworkSync import NetworkSyncToBehavior
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

    if len(sys.argv)>1:
        context.network_syncer.player_name = sys.argv[1]
    print(sys.argv)

    my_cam = CameraNode(parent=context.scene, name="MainCamera")
    my_cam.speed = 0.2
    my_cam.position = QVector3D(0, 5, 10)
    my_cam.add_behavior(Behavior.MoveAround(0.2))
    my_cam.add_behavior(Behavior.SnapToCamera())
    my_cam.ray = GeomNodes.OdeRayBehavior(obj=my_cam)
    my_cam.add_behavior(my_cam.ray)
    # my_cam.add_behavior(NetworkSyncToBehavior(my_cam))

    context.scene.context.current_camera = my_cam

    SkyBox(parent=context.current_camera)

    g = Grid(parent=context.scene)
    g.orientation = QQuaternion.fromEulerAngles(0.0, 0.0, 90.0)

    window.load_scene()

    create_button = CubeNode(parent=context.scene, name="CreateNotebookButton")
    create_button.tooltip = "Create New Notebook"
    create_button.events_handlers[Events.Clicked].append(lambda: create_new_notebook(context))
    create_button.scale = QVector3D(0.7, 0.2, 0.2)
    create_button.position = QVector3D(-3, 5, 0.2)
    create_button.on_event(Events.GeometryChanged)
    create_button.properties["skip serialization"] = True

    sph = Sphere(name="SpherePointer", parent=context.scene)
    sph.scale = QVector3D(0.05, 0.05, 0.05)
    sph.collider_id = None
    sph.visible = False
    context.debug_sphere = sph
    sph.properties["skip serialization"] = True

    window.show()
    window.setFocus()

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app.setQuitOnLastWindowClosed(True)
    ret = app.exec_()
    window.save_scene()
    sys.exit(ret)

