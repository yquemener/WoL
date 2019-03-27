#!/usr/bin/env python
import sys
import random
import signal
import os

from PyQt5.QtGui import QVector3D, QQuaternion, QMatrix4x4, QCursor
from PyQt5.QtWidgets import QApplication

from wol.GeomNodes import Grid, CardNode, Sphere
from wol.GuiNode import GuiNode
from wol.ConsoleNode import ConsoleNode
from wol.View3D import View3D


def camera_update(self, dt=0.0):
    speed = 0.2
    yaw = self.context.mouse_position.x() * 180.0
    pitch = -self.context.mouse_position.y() * 90.0

    x = QVector3D(1, 0, 0)
    yaw_rotation = QQuaternion.fromAxisAndAngle(0, 1, 0, yaw)
    pitch_axis = yaw_rotation.rotatedVector(x)
    pitch_rotation = QQuaternion.fromAxisAndAngle(pitch_axis, pitch)

    m = QMatrix4x4()
    m.rotate(pitch_rotation)
    m.rotate(yaw_rotation)
    self.context.current_camera.look_at = m * QVector3D(0, 0, 1)
    self.context.current_camera.up = QVector3D(0, 1, 0)
    right = QVector3D.crossProduct(self.context.current_camera.look_at,
                                   self.context.current_camera.up).normalized() * speed

    for action, delta in (
            ('forward', self.context.current_camera.look_at*speed),
            ('back',    -self.context.current_camera.look_at*speed),
            ('left',    -right),
            ('right',   right)):
        if self.context.abstract_input.get(action, False):
            self.context.current_camera.position += delta

    if self.context.abstract_input.get('active_action', False):
        print("action")

    self.sphere.position = self.context.debug_point


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = View3D()

    x = 0
    for fn in os.listdir("."):
        if not fn.endswith(".py"):
            continue
        o = GuiNode(parent=window.scene, name="GuiNode:" + str(fn), filename=fn)
        o.position = QVector3D(x, 2, x)
        x += 1
    x = 0
    for fn in os.listdir("wol/"):
        if not fn.endswith(".py"):
            continue
        o = GuiNode(parent=window.scene, name="GuiNode:" + str(fn), filename="wol/"+fn)
        o.position = QVector3D(10 + x, 2, x)
        x += 1

    o = ConsoleNode(parent=window.scene, name="ConsoleNode#1")
    o.position = QVector3D(0, 5, 0)

    g = Grid(parent=window.scene)
    g.orientation = QQuaternion.fromEulerAngles(0.0, 0.0, 90.0)
    sph = Sphere(parent=window.scene)
    window.scene.sphere = sph
    sph.size = 0.03
    # Monkey patching!
    window.scene.update = camera_update.__get__(window.scene)
    window.scene.context.current_camera.position = QVector3D(0, 5, -5)
    window.show()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(app.exec_())

