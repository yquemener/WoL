#!/usr/bin/env python
import sys
import signal
import os

from PyQt5.QtGui import QVector3D, QQuaternion, QMatrix4x4, QMatrix3x3
from PyQt5.QtWidgets import QApplication

from wol.GeomNodes import Grid, Sphere, CardNode
from wol.CodeBumperNode import CodeBumperNode
from wol.ConsoleNode import ConsoleNode
from wol.GuiElements import TextLabelNode
from wol.ObjectEditorNode import ObjectEditorNode
from wol.SceneNode import RootNode, CameraNode, SkyBox, SceneNode
from wol.Server import ServerNode
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

        self.socket.sendto(bytes(f"pos {self.position[0]} {self.position[1]} {self.position[2]}", "ascii"), ('localhost', 8971))

        """if self.context.grabbed is not None:
            cam = self.context.current_camera
            grab = self.context.grabbed
            m = self.context.grabbed_transform
            m = m * cam.transform
            m = m * grab.parent.transform.inverted()[0]
            wpos = m.map(QVector3D())
            dv = wpos - grab.world_position()
            grab.position += dv

            #print(self.context.grabbed.parent.transform.inverted()[0].map(wpos))
            #self.context.grabbed.position = self.context.grabbed.parent.transform.inverted()[0].map(wpos)
        """
        if not self.logged_in:
            try:
                self.socket.sendto(bytes("Hi! Yves", 'ascii'), ('localhost', 8971))
                self.logged_in = True
                print("Logged in")
            finally:
                pass



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = View3D()
    context = window.context

    load = False
    if load:
        try:
            lines = open("scene.ini").read().split("\n")
        except FileNotFoundError:
            lines = ()
            load = False
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
                # TODO: Make these kludges disappear
                if objtype == "Sphere":
                    context.scene.sphere = currentobj
                if objtype == "MyCamera":
                    context.scene.current_camera = currentobj
                    my_cam = currentobj
                objtype = None

            for r in reparenting:
                r[0].reparent(SceneNode.uid_map[r[1]], False)

    if not load:
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
        o.position = QVector3D(0, 5, -5)

        #o2 = ObjectEditorNode(parent=context.scene, target_object=o)
        #o2.position = QVector3D(0, 5, 0)

        o3 = CardNode(parent=context.scene, filename="resources/alphatest.png")
        o3.position = QVector3D(0, 10, 0)

        g = Grid(parent=context.scene)
        g.orientation = QQuaternion.fromEulerAngles(0.0, 0.0, 90.0)
        sph = Sphere(name="SpherePointer", parent=context.scene)
        context.scene.sphere = sph
        sph.scale = QVector3D(0.03, 0.03, 0.03)
        my_cam = MyCamera(context.scene)
        context.scene.context.current_camera = my_cam
        context.scene.context.current_camera.position = QVector3D(5, 5, 0)

        #o = TextEditNode(parent=context.scene, name="O_Node", filename="pieces/o_action.py")
        o = CodeBumperNode(parent=context.scene, name="CodeBumper#1", filename="pieces/cb1")
        o.position = QVector3D(0, 5, 8)

        sb = SkyBox(parent=my_cam)

        ser = ServerNode(parent=context.scene)
        ser.position = QVector3D(0, -1, 0)

    context.scene.context.current_camera = my_cam
    context.scene.context.current_camera.position = QVector3D(5, 5, 0)

    window.show()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(app.exec_())

