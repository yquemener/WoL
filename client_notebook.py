#!/usr/bin/env python

import sys
import signal

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QVector3D, QQuaternion

from wol import Behavior

from wol.Constants import UserActions
from wol.Notebook import NotebookNode
from wol.SceneNode import CameraNode, SkyBox
from wol.GeomNodes import Grid
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
        self.context.focused = None


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

    # ConsoleNode(parent=context.scene, name="ConsoleNode")
    context.hud_editor = HUDEditor(parent=my_cam, name="NotebookConsole")
    context.hud_editor.position = QVector3D(0, 0, 5)
    context.hud_editor.orientation = QQuaternion.fromEulerAngles(0.0, 180.0, 0.0)
    context.hud_editor.visible = False

    nb = NotebookNode(parent=context.scene, name="Notebook")
    # nb.orientation = QQuaternion.fromEulerAngles(0.0, 180.0, 0.0)
    nb.position = QVector3D(0, 5, 0)

    window.show()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(app.exec_())
