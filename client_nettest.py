#!/usr/bin/env python

import signal
import sys

from PyQt5.QtGui import QVector3D, QQuaternion
from PyQt5.QtWidgets import QApplication

from wol import Behavior, GeomNodes
from wol.Constants import Events
from wol.GeomNodes import Grid, Sphere, CubeNode
from wol.NetworkSync import NetworkSyncToBehavior
from wol.SceneNode import CameraNode, SkyBox
from wol.View3D import View3D

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = View3D()
    context = window.context

    if len(sys.argv) > 1:
        print(sys.argv)
        window.setGeometry(int(sys.argv[1]), 10, 800, 600)

    if len(sys.argv) > 2:
        context.network_syncer.player_name = sys.argv[2]
    print(sys.argv)

    context.network_syncer.connect()

    my_cam = CameraNode(parent=context.scene, name="MainCamera")
    my_cam.speed = 0.2
    my_cam.position = QVector3D(0, 5, 10)
    my_cam.add_behavior(Behavior.MoveAround(0.2))
    my_cam.add_behavior(Behavior.SnapToCamera())
    my_cam.ray = GeomNodes.OdeRayBehavior(obj=my_cam)
    my_cam.add_behavior(my_cam.ray)
    my_cam.add_behavior(NetworkSyncToBehavior(my_cam))

    context.scene.context.current_camera = my_cam

    SkyBox(parent=context.current_camera)

    g = Grid(parent=context.scene)
    g.orientation = QQuaternion.fromEulerAngles(0.0, 0.0, 90.0)

    sph = Sphere(name="SpherePointer", parent=context.scene)
    sph.scale = QVector3D(0.05, 0.05, 0.05)
    sph.collider_id = None
    # sph.visible = False
    context.debug_sphere = sph
    sph.properties["skip serialization"] = True

    window.show()
    window.setFocus()

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    ret = app.exec_()
    window.save_scene()
    sys.exit(ret)

