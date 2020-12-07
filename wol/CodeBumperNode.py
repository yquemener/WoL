""" Object that executes a python code when clicked or space-clicked and that opens a text
editor with the code when e-clicked. Hopefully also saves the code from one session to the other.
"""
import multiprocessing
import threading
import queue

import sys
from multiprocessing.queues import Queue

from PyQt5.QtGui import QVector3D, QVector4D
from io import StringIO

from wol import stdout_helpers
from wol.Behavior import Behavior, RotateConstantSpeed
from wol.GeomNodes import CubeNode, WireframeCubeNode
from wol.GuiElements import TextLabelNode
from wol.SceneNode import SceneNode
from wol.TextEditNode import TextEditNode

from random import random

from wol.utils import KillableThread


class StdoutQueue(Queue):
    def __init__(self, *args, **kwargs):
        ctx = multiprocessing.get_context()
        super(StdoutQueue, self).__init__(*args, **kwargs, ctx=ctx)

    def write(self, msg):
        self.put(msg)

    def flush(self):
        sys.__stdout__.flush()


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


class CodeRunnerEditorRunBehaviorRun(Behavior):
    def __init__(self, run_mode=1):
        super().__init__()
        self.run_mode = run_mode

    def on_click(self, evt, pos):
        self.obj.parent.run_code(self.run_mode)


class CodeRunnerEditorRunBehaviorKill(Behavior):
    def __init__(self):
        super().__init__()

    def on_click(self, evt, pos):
        # self.obj.parent.thread.kill()
        if self.obj.parent.process is not None:
            if self.obj.parent.process.is_alive():
                self.obj.parent.process.terminate()
        if self.obj.parent.thread is not None:
            self.obj.parent.thread.kill_me = True
        # self.obj.parent.redirected_output_thread = None
        # self.obj.parent.redirected_output_process = None
        # self.obj.parent.redirected_output_pos = 0


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

        self.button_run_process = CubeNode(parent=self, color=QVector4D(0, 1, 0, 0.5))
        self.button_run_process.position = QVector3D(1, 0.5, 0)
        self.button_run_process.scale = QVector3D(0.1, 0.1, 0.1)
        self.button_run_process.add_behavior(CodeRunnerEditorRunBehaviorRun(1))
        self.button_run_process.tooltip = "Run in a process"

        self.button_run_thread = CubeNode(parent=self, color=QVector4D(0, 1, 0, 0.5))
        self.button_run_thread.position = QVector3D(1.24, 0.5, 0)
        self.button_run_thread.scale = QVector3D(0.1, 0.1, 0.1)
        self.button_run_thread.add_behavior(CodeRunnerEditorRunBehaviorRun(2))
        self.button_run_thread.tooltip = "Run in a thread"

        self.button_kill = CubeNode(parent=self, color=QVector4D(0.3, 0.3, 0.3, 0.5))
        self.button_kill.scale = QVector3D(0.1, 0.1, 0.1)
        self.button_kill.position = QVector3D(1, 0.3, 0)
        self.button_kill.add_behavior(CodeRunnerEditorRunBehaviorKill())
        self.button_kill.tooltip = "Kill"

        self.run_indicator = WireframeCubeNode(parent=self, color=QVector4D(1, 1, 1, 1))
        self.run_indicator.scale = QVector3D(0.1, 0.1, 0.1)
        self.run_indicator.position = QVector3D(1, 1, 0)
        self.run_indicator.tooltip = "Not running"
        self.indicator_behavior = self.run_indicator.add_behavior(RotateConstantSpeed())

        self.text_edit = TextEditNode(parent=self,
                                      name=self.name+"_edit",
                                      text=text,
                                      autosize=True)
        self.text_edit.widget.textChanged.connect(self.on_text_changed)

        self.title_bar = TextLabelNode(name=self.name + "_titlebar", parent=self, text=filename)
        self.title_bar.position = QVector3D(0, 1, 0)

        self.output_text = TextLabelNode(name=self.name + "_output", parent=self, text="test ")
        self.output_text.position = QVector3D(0, -1, 0)

        self.process = None
        self.thread = None
        self.redirected_output_process = None
        self.redirected_output_thread = None
        self.redirected_output_pos = 0

        for c in self.children:
            c.properties["delegateGrabToParent"] = True

    def on_text_changed(self):
        f = open(self.filename, "w")
        f.write(self.text_edit.widget.toPlainText())
        f.close()

    def run_code(self, mode=1):
        self.output_text.set_text(" ")
        if mode == 1:
            self.redirected_output_process = StdoutQueue()
            self.process = multiprocessing.Process(target=self.threaded_func, args=(mode, self.redirected_output_process,))
            self.process.start()
        elif mode == 2:
            self.redirected_output_thread = StringIO()
            self.thread = KillableThread(target=self.threaded_func, args=(mode, self.redirected_output_process))
            self.thread.start()

    def threaded_func(self, mode, stdout_queue):
        if mode == 1:
            sys.stdout = stdout_queue
        elif mode == 2:
            self.redirected_output_thread = stdout_helpers.redirect()
        new_globals = globals()
        # new_locals = locals()
        # new_globals = dict()
        new_locals = dict()
        try:
            exec(self.text_edit.text, new_globals, new_locals)
        except KeyboardInterrupt:
            pass

    def update(self, dt):
        if self.redirected_output_process is not None:
            try:
                newtext = self.output_text.text
                newtext += self.redirected_output_process.get(False)
                self.output_text.set_text(newtext)
            except queue.Empty:
                pass

        if self.redirected_output_thread is not None:
            if self.redirected_output_pos != self.redirected_output_thread.tell():
                self.redirected_output_pos = self.redirected_output_thread.tell()
                self.output_text.set_text(self.redirected_output_thread.getvalue())

        if (self.process is not None and self.process.is_alive()) or \
                (self.thread is not None and self.thread.is_alive()):
            self.indicator_behavior.speed = 200
            self.run_indicator.tooltip = "Running"

        else:
            self.indicator_behavior.speed = 0
            self.run_indicator.tooltip = "Not running"





