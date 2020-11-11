""" Object that executes a python code when clicked or space-clicked and that opens a text
editor with the code when e-clicked. Hopefully also saves the code from one session to the other.
"""
from PyQt5.QtGui import QVector3D, QVector4D

from wol.Behavior import Behavior, RotateConstantSpeed
from wol.GeomNodes import CubeNode, WireframeCubeNode
from wol.GuiElements import TextLabelNode
from wol.SceneNode import SceneNode
from wol.TextEditNode import TextEditNode

import threading
from random import random


class CodeBumperNode(TextLabelNode):
    def __init__(self, parent=None, name="CodeBumber", label="BUMP", filename=None, code=" "):
        TextLabelNode.__init__(self, text=label, name=name, parent=parent)
        if filename is not None:
            self.filename = filename
            try:
                text = open(filename).read()
            except FileNotFoundError:
                text = " "
        else:
            text = code
            self.filename = None

        self.edit_node = TextEditNode(parent=self,
                                      name=self.name+"#edit",
                                      text=text,
                                      autosize=True)
        self.edit_node.visible = False
        self.edit_node.position = QVector3D(1.2, 0, random()*0.1-0.05)
        """for v in self.vertices:
            v[0] *= 0.2
            v[1] *= 0.2
            v[2] *= 0.2"""
        self.refresh_vertices()
        self.locals = dict(locals())

    def on_click(self, pos, evt):
        code = self.edit_node.widget.toPlainText()
        try:
            exec(code, globals(), self.locals)
        except Exception as e:
            print("Exception: " + str(e) + "\n")

    def on_edit(self, pos):
        self.edit_node.visible = not self.edit_node.visible
        if self.filename is not None:
            f = open(self.filename, "w")
            f.write(self.edit_node.text)
            f.close()


class CodeRunnerEditorRunBehavior(Behavior):
    def __init__(self):
        super().__init__()

    def on_click(self, evt, pos):
        self.obj.parent.run_code()


class CodeRunnerEditorNode(SceneNode):
    def __init__(self, name="codeRunnerEditor", parent=None, filename=None):
        super().__init__(name, parent=parent)
        if filename is None:
            raise ValueError("You have to specify a textfile")
        self.filename = filename
        try:
            text = open(filename).read()
        except FileNotFoundError:
            text = " "

        self.button_run = CubeNode(parent=self, color=QVector4D(0, 1, 0, 0.5))
        self.button_run.position = QVector3D(1, 0.5, 0)
        self.button_run.scale = QVector3D(0.1, 0.1, 0.1)
        self.button_run.add_behavior(CodeRunnerEditorRunBehavior())

        self.button_kill = CubeNode(parent=self, color=QVector4D(0.3, 0.3, 0.3, 0.5))
        self.button_kill.scale = QVector3D(0.1, 0.1, 0.1)
        self.button_kill.position = QVector3D(1, 0.3, 0)

        self.run_indicator = WireframeCubeNode(parent=self, color=QVector4D(1, 1, 1, 1))
        self.run_indicator.scale = QVector3D(0.1, 0.1, 0.1)
        self.run_indicator.position = QVector3D(1, 1, 0)
        self.indicator_behavior = self.run_indicator.add_behavior(RotateConstantSpeed())

        self.text_edit = TextEditNode(parent=self,
                                      name=self.name+"_edit",
                                      text=text,
                                      autosize=True)
        self.title_bar = TextLabelNode(name=self.name + "_titlebar", parent=self, text=filename)
        self.title_bar.position = QVector3D(0, 1, 0)

        self.thread = None

        for c in self.children:
            c.properties["delegateGrabToParent"] = True

    def run_code(self):
        self.thread = threading.Thread(target=self.threaded_func)
        self.thread.start()
        pass

    def threaded_func(self):
        # redirected_output = sys.stdout = StringIO()
        exec(self.text_edit.text, dict(), dict())

    def update(self, dt):
        if self.thread is not None and self.thread.is_alive():
            self.indicator_behavior.speed = 200
        else:
            self.indicator_behavior.speed = 0




