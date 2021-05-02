#!/usr/bin/env python

import sys
import signal

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QVector3D, QQuaternion

from wol import Behavior
from wol.ConsoleNode import ConsoleNode
from wol.SceneNode import CameraNode, SkyBox, SceneNode
from wol.GeomNodes import Grid
from wol.TextEditNode import TextEditNode
from wol.View3D import View3D


class HUDEditor(TextEditNode):
    def __init__(self, parent, name):
        TextEditNode.__init__(self, parent=parent, name=name)

    def keyPressEvent(self, evt):
        if evt.key() == QtCore.Qt.Key_Return and \
            QApplication.keyboardModifiers() == QtCore.Qt.ShiftModifier:
            exec(self.text, self.context.execution_context)
        super().keyPressEvent(evt)

        # if evt.key() == Qt.Key_Return and QApplication.keyboardModifiers() == Qt.KeyBoardmodifiers


class NotebookNode(SceneNode):
    def __init__(self, parent, name):
        SceneNode.__init__(self, parent=parent, name=name)
        cell = TextEditNode(parent=self, name=name+"_cell1")
        self.add_child(cell)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = View3D()
    context = window.context

    my_cam = CameraNode(parent=context.scene, name="MainCamera")
    my_cam.speed = 0.2
    my_cam.position = QVector3D(5, 5, -10)
    my_cam.add_behavior(Behavior.MoveAround(0.2))
    my_cam.add_behavior(Behavior.SnapToCamera())
    context.scene.context.current_camera = my_cam

    SkyBox(parent=context.current_camera)

    g = Grid(parent=context.scene)
    g.orientation = QQuaternion.fromEulerAngles(0.0, 0.0, 90.0)

    # ConsoleNode(parent=context.scene, name="ConsoleNode")
    o = HUDEditor(parent=my_cam, name="NotebookConsole")
    o.position = QVector3D(0,0,10)
    o.visible = False

    NotebookNode(parent=context.scene, name="Notebook")

    window.show()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(app.exec_())
