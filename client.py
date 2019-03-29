#!/usr/bin/env python
import sys
import signal
import os

from PyQt5.QtGui import QVector3D, QQuaternion, QMatrix4x4
from PyQt5.QtWidgets import QApplication

from wol.GeomNodes import Grid, Sphere
from wol.CodeBumperNode import CodeBumperNode
from wol.ConsoleNode import ConsoleNode
from wol.SceneNode import CameraNode
from wol.TextEditNode import TextEditNode
from wol.View3D import View3D


class MyCamera(CameraNode):
    def __init__(self, parent):
        CameraNode.__init__(self, parent=parent)
        self.speed = 0.2

    def update(self, dt):
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
                                       self.context.current_camera.up).normalized() * self.speed

        for action, delta in (
                ('forward', self.context.current_camera.look_at*self.speed),
                ('back',    -self.context.current_camera.look_at*self.speed),
                ('left',    -right),
                ('right',   right)):
            if self.context.abstract_input.get(action, False):
                self.context.current_camera.position += delta
        self.context.scene.sphere.position = self.context.debug_point


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = View3D()
    context = window.context

    x = 0
    for fn in os.listdir("pieces"):
        o = TextEditNode(parent=context.scene, name="GuiNode:" + str(fn), filename="pieces/"+fn)
        o.position = QVector3D(x, 2, x)
        x += 1
    x = 0
    for fn in os.listdir("wol/"):
        if not fn.endswith(".py"):
            continue
        o = TextEditNode(parent=context.scene, name="GuiNode:" + str(fn), filename="wol/"+fn)
        o.position = QVector3D(10 + x, 2, x)
        x += 1

    o = ConsoleNode(parent=context.scene, name="ConsoleNode#1")
    o.position = QVector3D(0, 5, -3)

    o = CodeBumperNode(parent=context.scene, name="CodeBumper#1")
    o.position = QVector3D(0, 5, 0)

    g = Grid(parent=context.scene)
    g.orientation = QQuaternion.fromEulerAngles(0.0, 0.0, 90.0)
    sph = Sphere(name="SpherePointer", parent=context.scene)
    context.scene.sphere = sph
    sph.size = 0.03
    # Monkey patching!
    my_cam = MyCamera(context.scene)
    context.scene.context.current_camera = my_cam
    #context.scene.update = camera_update.__get__(context.scene)
    context.scene.context.current_camera.position = QVector3D(0, 5, -5)
    window.show()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(app.exec_())

