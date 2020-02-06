#!/usr/bin/env python
import sys
import signal
import os

import time

import math
from PyQt5.QtGui import QVector3D, QQuaternion, QMatrix4x4, QVector4D
from PyQt5.QtWidgets import QApplication

from wol import Behavior
from wol.GeomNodes import Grid, Sphere, CardNode, WireframeCubeNode, CubeNode
from wol.CodeBumperNode import CodeBumperNode
from wol.ConsoleNode import ConsoleNode
from wol.ObjectEditorNode import ObjectEditorNode
from wol.PythonFileEditorNode import PythonFileEditorNode
from wol.SceneNode import CameraNode, SkyBox, SceneNode
from wol.Server import ServerNode, ClientNode
from wol.TextEditNode import TextEditNode
from wol.View3D import View3D
import socket


class MyCamera(CameraNode):
    def __init__(self, parent, name="MyCam"):
        CameraNode.__init__(self, parent=parent, name=name)
        self.speed = 0.2

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.logged_in = False

    def update(self, dt):
        yaw = self.context.mouse_position.x() * 180.0
        pitch = -self.context.mouse_position.y() * 90.0

        xaxis = QVector3D(1, 0, 0)
        yaw_rotation = QQuaternion.fromAxisAndAngle(0, 1, 0, yaw)
        pitch_axis = yaw_rotation.rotatedVector(xaxis)
        pitch_rotation = QQuaternion.fromAxisAndAngle(pitch_axis, pitch)

        m = QMatrix4x4()
        m.rotate(pitch_rotation)
        m.rotate(yaw_rotation)
        direction = m * QVector3D(0, 0, 1)
        self.context.current_camera.look_at = self.context.current_camera.position + direction
        self.context.current_camera.up = QVector3D(0, 1, 0)
        self.orientation = pitch_rotation * yaw_rotation
        right = QVector3D.crossProduct(direction,
                                       self.context.current_camera.up).normalized() * self.speed

        for action, delta in (
                ('forward', direction*self.speed),
                ('back',    -direction*self.speed),
                ('left',    -right),
                ('right',   right),
                ('up', self.context.current_camera.up * self.speed),
                ('down', -self.context.current_camera.up * self.speed)):
            if self.context.abstract_input.get(action, False):
                self.context.current_camera.position += delta
                self.context.current_camera.look_at += delta

        """self.socket.sendto(
            bytes(f"pos {self.position[0]} {self.position[1]} {self.position[2]}", "ascii"), 
            ('localhost', 8971))

        if not self.logged_in:
            try:
                self.socket.sendto(bytes("Hi! Yves", 'ascii'), ('localhost', 8971))
                self.logged_in = True
                print("Logged in")
            finally:
                pass"""


class B1(Behavior.Behavior):
    def on_update(self, dt):
        self.obj.position.setX(math.cos(time.time()))


# def make_slerp_anim(delay = 1.0):
#     def slerp_anim(obj, dt):
#         obj.anim_timer += dt
#         o1 = obj.step1.world_orientation()
#         o2 = obj.step2.world_orientation()
#         p1 = obj.step1.world_position()
#         p2 = obj.step2.world_position()
#
#         alpha = abs(obj.anim_timer / delay)
#         if alpha > 1.0:
#             alpha = 1.0
#         obj.position = p2*alpha + p1*(1.0-alpha)
#         obj.orientation = QQuaternion.slerp(o1, o2, alpha)
#         # obj.position = p1
#         # obj.orientation = obj.step1.world_orientation()
#     return slerp_anim


def load_scene_ini():
    lines = open("scene.ini").read().split("\n")
    attributes = dict()
    objtype = None
    currentobj = None
    reparenting = list()
    for line in lines:
        line = line.lstrip()
        if line.startswith("new "):
            objtype = line.split(" ")[1]
            attributes = dict()
            attributes['parent'] = context.scene
        if "=" in line:
            name, value = line.split("=")
            attributes[name] = value
        if line.startswith("move "):
            currentobj.position = QVector3D(*[float(x) for x in line.split(" ")[1:]])
        if line.startswith("scale "):
            currentobj.scale = QVector3D(*[float(x) for x in line.split(" ")[1:]])
        if line.startswith("rotate "):
            currentobj.orientation = QQuaternion(*[float(x) for x in line.split(" ")[1:]])
        if line.startswith("setuid "):
            currentobj.set_uid(int(line.split(" ")[1]))
        if line.startswith("setparentuid "):
            reparenting.append((currentobj, int(line.split(" ")[1])))
        if line.rstrip().lstrip() == "" and objtype is not None:
            currentobj = globals()[objtype](**attributes)
            # TODO: Make this kludge disappear
            if objtype == "MyCamera":
                context.scene.current_camera = currentobj
                my_cam = currentobj
            objtype = None

        for r in reparenting:
            r[0].reparent(SceneNode.uid_map[r[1]], False)
    return my_cam


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = View3D()
    context = window.context

    load = False
    if load:
        try:
            my_cam = load_scene_ini()
        except FileNotFoundError:
            load = False
    if not load:

        o = ConsoleNode(parent=context.scene, name="ConsoleNode#1")
        o.position = QVector3D(0, 5, -5)

        g = Grid(parent=context.scene)
        g.orientation = QQuaternion.fromEulerAngles(0.0, 0.0, 90.0)
        sph = Sphere(name="SpherePointer", parent=context.scene)
        sph.scale = QVector3D(0.03, 0.03, 0.03)
        my_cam = MyCamera(context.scene)
        context.scene.context.current_camera = my_cam
        context.scene.context.current_camera.position = QVector3D(5, 5, 0)

        # o3 = CardNode(parent=my_cam, filename="resources/alphatest.png")
        # o3.position = QVector3D(0, 0, 10)

        o4 = SceneNode(parent=context.scene)
        o4.position = QVector3D(0, 0, -10)
        o4.orientation = QQuaternion.fromAxisAndAngle(1, 0, 0, 90)

        # o5 = CardNode(parent=context.scene, filename="resources/alphatest.png")
        # o5.anim_timer = 0.0
        # o5.step1 = o3
        # o5.step2 = o4
        # o5.behaviors.append(make_slerp_anim(10.0))

        o = CodeBumperNode(parent=context.scene, name="CodeBumper#1", filename="pieces/cb1")
        o.position = QVector3D(0, 5, 8)

        sb = SkyBox(parent=my_cam)

        """serPar = SceneNode(name="ServerParent", parent=context.scene)
        serPar.position = QVector3D(0, 4, 0)

        serSph = Sphere(name="SpherePointer", parent=serPar)
        serSph.scale = QVector3D(0.2, 0.2, 0.2)
        serSph.position = QVector3D(0, 0.4, 0)
        serSph.properties["delegateGrabToParent"] = True"""

        ser = ServerNode(parent=context.scene)
        ser.position = QVector3D(0, 4, 0)
        context.server = ser

        cli = ClientNode(parent=context.scene)
        cli.position = QVector3D(3, 4, 0)

        objed = PythonFileEditorNode(parent=context.scene, target_file_name="wol/SceneNode.py")
        objed.position = QVector3D(10, 2, 0)

        wcube = WireframeCubeNode(parent=context.scene, color=QVector4D(255,255,255,255))
        wcube.position = QVector3D(10, 3, 0)
        wcube.scale = QVector3D(0.1, 0.1, 0.1)

        cube = CubeNode(parent=context.scene, color=QVector4D(255,255,255,255))
        cube.position = QVector3D(10, 4, 0)
        cube.scale = QVector3D(0.1, 0.1, 0.1)

    my_cam.add_behavior(Behavior.SnapToCamera())
    context.scene.context.current_camera = my_cam
    context.scene.context.current_camera.position = QVector3D(5, 5, 0)

    window.show()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(app.exec_())


