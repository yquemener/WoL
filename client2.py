#!/usr/bin/env python

import signal
import socket
import sys
import threading

from PyQt5.QtGui import QVector3D, QQuaternion
from PyQt5.QtWidgets import QApplication

from wol import Behavior, DevScenes, stdout_helpers, ConsoleNode, GeomNodes
from wol.GuiElements import CodeSnippetReceiver, CodeSnippet
from wol.NetworkSync import NetworkSyncer, NetworkSyncToBehavior
from wol.SceneNode import CameraNode, SkyBox
from wol.View3D import View3D
import wol.protobuf.message_pb2 as protocol


class MyCamera(CameraNode):
    def __init__(self, parent, name="MyCam"):
        CameraNode.__init__(self, parent=parent, name=name)
        self.speed = 0.2
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.logged_in = False
        self.add_behavior(Behavior.MoveAround(0.2))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = View3D()
    context = window.context
    stdout_helpers.enable_proxy()

    my_cam = MyCamera(context.scene)
    context.scene.context.current_camera = my_cam
    my_cam.add_behavior(Behavior.SnapToCamera())
    my_cam.add_behavior(ConsoleNode.InvokeConsole())
    my_cam.ray = GeomNodes.OdeRayBehavior(obj=my_cam)
    context.scene.context.current_camera = my_cam
    SkyBox(parent=context.scene.context.current_camera)

    context.network_syncer = NetworkSyncer()

    if len(sys.argv) > 1:
        print(sys.argv)
        window.setGeometry(int(sys.argv[1]), 10, 800, 600)

    if len(sys.argv) > 2:
        context.network_syncer.player_name = sys.argv[2]
    print(sys.argv)

    context.network_syncer.connect()


    sph = GeomNodes.Sphere(name="SpherePointer", parent=context.scene)
    sph.scale = QVector3D(0.05, 0.05, 0.05)
    sph.collider_id = None
    # sph.visible = False
    context.debug_sphere = sph

    g = GeomNodes.Grid(parent=context.scene)
    g.orientation = QQuaternion.fromEulerAngles(0.0, 0.0, 90.0)

    snip = CodeSnippetReceiver(parent=context.scene)
    snip.position = QVector3D(0, 4, 0)
    snip.orientation = QQuaternion.fromEulerAngles(0, 180, 0)

    snip = CodeSnippet(parent=context.scene)
    snip.position = QVector3D(4, 4, 0)
    snip.set_text('print("Hello world")')
    snip.orientation = QQuaternion.fromEulerAngles(0, 180, 0)
    snip.add_behavior(NetworkSyncToBehavior(obj=snip))

    context.scene.context.current_camera.position = QVector3D(5, 5, -10)

    window.show()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(app.exec_())







































