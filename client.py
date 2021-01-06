#!/usr/bin/env python
import signal
import socket
import sys

from PyQt5.QtGui import QVector3D, QQuaternion, QMatrix4x4
from PyQt5.QtWidgets import QApplication

from wol import Behavior, DevScenes, stdout_helpers
from wol.SceneNode import CameraNode, SceneNode, SkyBox
from wol.View3D import View3D


class MyCamera(CameraNode):
    def __init__(self, parent, name="MyCam"):
        CameraNode.__init__(self, parent=parent, name=name)
        self.speed = 0.2
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.logged_in = False
        self.add_behavior(Behavior.MoveAround(0.2))


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
    stdout_helpers.enable_proxy()

    my_cam = MyCamera(context.scene)
    context.scene.context.current_camera = my_cam
    my_cam.add_behavior(Behavior.SnapToCamera())
    context.scene.context.current_camera = my_cam
    SkyBox(parent=context.scene.context.current_camera)

    load = True
    if load:
        try:
            window.load_scene()
            # my_cam = load_scene_ini()
        except FileNotFoundError:
            load = False
    if not load:

        DevScenes.scene_base(context)
        # DevScenes.scene_load(context)
        # DevScenes.scene_network(context)
        DevScenes.scene_ide(context)
        # DevScenes.scene_tests(context)
        # DevScenes.scene_gui_test(context)

    # context.scene.context.current_camera.position = QVector3D(5, 5, 0)
    context.scene.context.current_camera.position = QVector3D(5, 5, -10)
    window.show()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(app.exec_())
