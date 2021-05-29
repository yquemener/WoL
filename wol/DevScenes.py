# Just a place to store debug scenes as long as we don't have a good serialization of scenes
from PyQt5.QtGui import QVector3D, QQuaternion, QVector4D

from client import MyCamera
from wol import Behavior
from wol.CodeEdit import CodeBumperNode, CodeRunnerEditorNode, FileCodeNode
from wol.ConsoleNode import ConsoleNode
from wol.GeomNodes import Grid, Sphere, WireframeCubeNode, CubeNode, CardNode
from wol.GuiElements import WidgetTestNode
from wol.PythonFileEditorNode import PythonFileEditorNode
from wol.SceneNode import SceneNode, SkyBox
from wol.Server import ServerNode, ClientNode
from wol.archive.ObjectEditorNode import ObjectEditorNode


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
    context.debug_sphere = sph

def scene_base(context):
    SkyBox(parent=context.scene.context.current_camera)
    g = Grid(parent=context.scene)
    g.orientation = QQuaternion.fromEulerAngles(0.0, 0.0, 90.0)


def scene_load(context):
    fout = open("init_scene.py", "r")
    s = fout.read()
    fout.close()
    d = globals()
    d["context"] = context
    exec(s, d, d)


def scene_ide(context):
    context.scene.context.current_camera.position = QVector3D(5, 5, -10)

    o = ConsoleNode(parent=context.scene, name="ConsoleNode#1")
    o.position = QVector3D(0, 5, -5)
    context.current_console = o

    obj1 = ObjectEditorNode(parent=context.scene, target_object=o)
    obj1.position = QVector3D(0, 2, 5)

    # objed = PythonFileEditorNode(parent=context.scene, target_file_name="my_project/main.py")
    # objed.position = QVector3D(0, 2, 5)
    #
    # objed = FileCodeNode(parent=context.scene, filename="my_project/test.py")
    # objed.position = QVector3D(3, 4, -5)
    # objed.orientation = QQuaternion.fromEulerAngles(0, 180, 0)
    #
    # objed = FileCodeNode(parent=context.scene, filename="my_project/torchtest.py")
    # objed.position = QVector3D(10, 4, -5)
    # objed.orientation = QQuaternion.fromEulerAngles(0, 180, 0)
    #
    # objed = FileCodeNode(parent=context.scene, filename="my_project/main.py")
    # objed.position = QVector3D(5, 4, -5)
    # objed.orientation = QQuaternion.fromEulerAngles(0, 180, 0)
    #
    # objed = FileCodeNode(parent=context.scene, filename="my_project/server.py")
    # objed.position = QVector3D(8, 4, -5)
    # objed.orientation = QQuaternion.fromEulerAngles(0, 230, 0)
    #
    # objed = WidgetTestNode(parent=context.scene, text="my_project\nserver.py")
    # objed.position = QVector3D(5, 0, -5)
    # objed.orientation = QQuaternion.fromEulerAngles(-90, 230, 0)

    context.scene.context.debug_sphere = Sphere(name="SpherePointer", parent=context.scene)
    context.scene.context.debug_sphere.scale = QVector3D(0.1, 0.1, 0.1)
    context.scene.context.debug_sphere.collider = None


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


def scene_gui_test(context):
    context.scene.context.current_camera.position = QVector3D(5, 5, -10)

    objed = WidgetTestNode(parent=context.scene, text="my_project\nserver.py")
    objed.position = QVector3D(3, 4, -5)
    objed.orientation = QQuaternion.fromEulerAngles(0, 180, 0)
