#!/usr/bin/env python
import sys
import random
import signal

from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QVector3D, QQuaternion, QMatrix4x4, QCursor
from PyQt5.QtWidgets import QApplication

from wol.GeomNodes import Grid, CardNode, Sphere
from wol.GuiNode import GuiNode
from wol.View3D import View3D


def camera_update(self, dt=0.0):
    speed = 0.05
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

    for k in range(40):
        #o = CardNode(filename="/home/yves/path4787.png", parent=window.scene, name="Card#"+str(k))
        o = GuiNode(parent=window.scene, name="GuiNode#" + str(k))
        o.position = QVector3D(random.uniform(-1.0, 1.0),
                               random.uniform(-1.0, 1.0),
                               random.uniform(1.0, 30.0))
        # o.position = QVector3D(0.0, 0.0, -5.0)
    g = Grid(parent=window.scene)
    g.orientation = QQuaternion.fromEulerAngles(0.0, 0.0, 90.0)
    sph = Sphere(parent=window.scene)
    window.scene.sphere = sph
    sph.size = 0.03
    # Monkey patching!
    window.scene.update = camera_update.__get__(window.scene)
    window.show()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(app.exec_())
