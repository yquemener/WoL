#!/usr/bin/env python

import sys
import signal

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QVector3D, QQuaternion

from wol import Behavior
from wol.Constants import UserActions, Events
from wol.SceneNode import CameraNode, SkyBox, SceneNode
from wol.GeomNodes import Grid
from wol.TextEditNode import TextEditNode
from wol.View3D import View3D


class HUDEditor(TextEditNode):
    def __init__(self, parent, name):
        TextEditNode.__init__(self, parent=parent, name=name)
        self.target_object = None

    def keyPressEvent(self, evt):
        if evt.key() == QtCore.Qt.Key_Return and \
            QApplication.keyboardModifiers() == QtCore.Qt.ShiftModifier:
            self.visible = False
            self.context.focused = None
            self.target_object.set_text(self.text)
            self.target_object.on_event(UserActions.Execute)
        else:
            super().keyPressEvent(evt)
            self.needs_refresh = True
            self.target_object.keyPressEvent(evt)
            self.target_object.needs_refresh = True


class EditOnClick(Behavior.Behavior):
    def __init__(self):
        super().__init__()
        self.events_handlers[Events.Clicked].append(self.on_edit)
        self.focused = None

    def on_edit(self):
        ctxt = self.obj.context
        ctxt.hud_editor.visible = True
        ctxt.hud_editor.target_object = self.obj
        ctxt.focused = ctxt.hud_editor
        ctxt.hud_editor.set_text(self.obj.text)


class ExecuteBehavior(Behavior.Behavior):
    def __init__(self):
        super().__init__()
        self.events_handlers[UserActions.Execute].append(self.on_execute)

    def on_execute(self):
        exec(self.obj.text, self.obj.context.execution_context)


class NotebookNode(SceneNode):
    def __init__(self, parent, name):
        SceneNode.__init__(self, parent=parent, name=name)
        self.filename = f"my_project/{self.name}.py"
        self.cells = list()
        self.events_handlers[Events.AppClose].append(self.save)
        self.events_handlers[UserActions.Save].append(self.save)

        try:
            s = open(self.filename).read()
            sep = "\n" + "#**" * 20 + "\n"
            last_y = 0
            for i, celltext in enumerate(s.split(sep)):
                cell = TextEditNode(parent=self, name=name+"_cell"+str(i))
                cell.set_text(celltext)
                cell.add_behavior(EditOnClick())
                cell.add_behavior(ExecuteBehavior())
                cell.autosize = True
                cell.min_size = (200, 30)
                cell.position = QVector3D(0, last_y, 0)
                last_y -= 0.2
                self.add_child(cell)
                self.cells.append(cell)
        except FileNotFoundError:
            cell = TextEditNode(parent=self, name=name+"_cell1")
            cell.add_behavior(EditOnClick())
            cell.add_behavior(ExecuteBehavior())
            cell.autosize = True
            cell.min_size = (200, 30)
            self.add_child(cell)
            self.cells.append(cell)

    def update(self, dt):
        last_cell = self.cells[-1]
        if last_cell.text != "":
            cell = TextEditNode(parent=self, name=self.name + "_cell"+str(len(self.cells)+1))
            cell.add_behavior(EditOnClick())
            cell.add_behavior(ExecuteBehavior())
            cell.autosize = True
            cell.min_size = (200, 30)
            cell.position = QVector3D(last_cell.position)
            # cell.position.setY(cell.position.y() + last_cell.widget.size().height())
            cell.position.setY(cell.position.y() - 0.2)
            self.add_child(cell)
            self.cells.append(cell)

    def save(self):
        # Use notebook formats?
        f = open(self.filename, "w")
        sep = "\n" + "#**"*20 + "\n"
        f.write(sep.join([c.text for c in self.cells]))
        f.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = View3D()
    context = window.context

    my_cam = CameraNode(parent=context.scene, name="MainCamera")
    my_cam.speed = 0.2
    my_cam.position = QVector3D(5, 5, -10)
    my_cam.add_behavior(Behavior.MoveAround(0.2))
    my_cam.add_behavior(Behavior.SnapToCamera())
    context.scene.context.current_camera = my_cam

    SkyBox(parent=context.current_camera)

    g = Grid(parent=context.scene)
    g.orientation = QQuaternion.fromEulerAngles(0.0, 0.0, 90.0)

    # ConsoleNode(parent=context.scene, name="ConsoleNode")
    context.hud_editor = HUDEditor(parent=my_cam, name="NotebookConsole")
    context.hud_editor.position = QVector3D(0, 0, 5)
    context.hud_editor.orientation = QQuaternion.fromEulerAngles(0.0, 180.0, 0.0)
    context.hud_editor.visible = False

    nb = NotebookNode(parent=context.scene, name="Notebook")
    nb.orientation = QQuaternion.fromEulerAngles(0.0, 180.0, 0.0)
    nb.position = QVector3D(0, 5, 0)

    window.show()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(app.exec_())
