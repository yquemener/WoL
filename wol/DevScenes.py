# Just a place to store debug scenes as long as we don't have a good serialization of scenes
from PyQt5.QtGui import QVector3D, QQuaternion, QVector4D

from client import MyCamera
from wol import Behavior
from wol.CodeBumperNode import CodeBumperNode
from wol.ConsoleNode import ConsoleNode
from wol.GeomNodes import Grid, Sphere, WireframeCubeNode, CubeNode, CardNode
from wol.PythonFileEditorNode import PythonFileEditorNode
from wol.SceneNode import SceneNode, SkyBox
from wol.Server import ServerNode, ClientNode


def scene_tests(context):

    o5 = CardNode(parent=context.scene, filename="resources/alphatest.png")
    o5.anim_timer = 0.0
    o5.add_behavior(Behavior.SlerpAnim(10.0))

    wcube = WireframeCubeNode(parent=context.scene, color=QVector4D(1, 1, 1, 1))
    wcube.position = QVector3D(10, 2, 0)
    # wcube.scale = QVector3D(0.1, 0.1, 0.1)
    wcube.add_behavior(Behavior.RotateConstantSpeed(50.0))

    cube = CubeNode(parent=context.scene, color=QVector4D(1, 1, 1, 0.5))
    cube.position = QVector3D(10, 5, 0)
    # cube.scale = QVector3D(0.1, 0.1, 0.1)
    cube.add_behavior(Behavior.RotateConstantSpeed(50.0))

    sph = Sphere(name="SpherePointer", parent=context.scene)
    sph.scale = QVector3D(0.2, 0.2, 0.2)
    sph.collider = None
    context.debug_sphere = sph


def scene_base(context):
    g = Grid(parent=context.scene)
    g.orientation = QQuaternion.fromEulerAngles(0.0, 0.0, 90.0)
    context.scene.context.current_camera.position = QVector3D(5, 5, 0)

    SkyBox(parent=context.scene.context.current_camera)


def scene_ide(context):
    o = ConsoleNode(parent=context.scene, name="ConsoleNode#1")
    o.position = QVector3D(0, 5, -5)

    objed = PythonFileEditorNode(parent=context.scene, target_file_name="my_project/main.py")
    objed.position = QVector3D(0, 2, 5)


def scene_network(context):
    o = CodeBumperNode(parent=context.scene, name="CodeBumper#1", filename="pieces/cb1")
    o.position = QVector3D(0, 5, 8)

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

