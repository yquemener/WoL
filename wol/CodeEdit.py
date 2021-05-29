""" Object that executes a python code when clicked or space-clicked and that opens a text
editor with the code when e-clicked. Hopefully also saves the code from one session to the other.
"""
import multiprocessing
import threading
import queue

import sys
import time
from multiprocessing.queues import Queue
from traceback import TracebackException

import numpy
import pybullet
from OpenGL import GL
from PyQt5.QtGui import QVector3D, QVector4D, QQuaternion, QMatrix4x4, QImage
from io import StringIO

from wol import stdout_helpers
from wol.Behavior import Behavior, RotateConstantSpeed
from wol.Constants import UserActions, Events
from wol.GeomNodes import CubeNode, WireframeCubeNode, CardNode
from wol.GuiElements import TextLabelNode
from wol.SceneNode import SceneNode
from wol.ShadersLibrary import ShadersLibrary
from wol.TextEditNode import TextEditNode

from random import random

from wol.utils import KillableThread


def watch(obj, period=0):
    renderlist = globals().get("_wol_render_list", list())
    renderlist.append((obj, period))
    globals()["_wol_render_list"] = renderlist


class DataViewer(SceneNode):
    def __init__(self, parent, target, period):
        super().__init__(parent=parent)
        self.target = target
        self.period = period
        self.type_label = TextLabelNode(parent=self, text=str(target))
        self.type_label.position = QVector3D(0, self.type_label.hscale,0)
        self.content_view_text = TextLabelNode(parent=self, text="")
        self.content_view_text.position += QVector3D(0, -0.02, 0)
        self.content_view_image = CardNode(parent=self)
        self.content_view_image.position += QVector3D(0, -0.02, 0)
        self.content_view_image.visible = False
        self.content_view_image.interpolation = GL.GL_NEAREST
        for c in self.children:
            c.properties["delegateGrabToParent"] = True

        self.button_close = CubeNode(parent=self, color=QVector4D(0.8, 0.2, 0.2, 0.8))
        self.button_close.position = QVector3D(self.type_label.wscale+0.1, self.type_label.hscale, 0)
        self.button_close.scale = QVector3D(0.07, 0.07, 0.07)
        self.button_close.events_handlers[Events.Clicked].append(lambda:  self.remove())
        self.button_close.tooltip = "Close"

        self.vertices = list()
        self.program = None
        self.refresh_vertices()
        self.last_update = 0
        self.refresh_content()
        self.serialization_atributes.append("target")
        self.serialization_atributes.append("period")

    def refresh_content(self):
        try:
            target_obj = eval(self.target, self.context.execution_context)
        except Exception as e:
            target_obj = ""
            (_, value, tb) = sys.exc_info()
            for line in TracebackException(
                    type(value), value, tb, limit=None).format(chain=True):
                target_obj += line
        if isinstance(target_obj, str):
            self.content_view_text.set_text(str(target_obj))
            self.content_view_image.visible = False
            self.content_view_text.position = QVector3D(0,-0.02-self.content_view_text.hscale,0)
        elif isinstance(target_obj, numpy.ndarray):
            im = numpy.require(target_obj, numpy.uint8, 'C')
            if len(im.shape) > 2 and im.shape[2] == 3:
                fmt = QImage.Format_RGB888
            else:
                fmt = QImage.Format_Grayscale8
            self.content_view_image.texture_image = \
                QImage(im.data, im.shape[1], im.shape[0], im.strides[0], fmt).copy()
            self.content_view_image.initialize_gl()
            self.content_view_image.visible = True
            # self.content_view_text.set_text(str(target_obj))
            self.content_view_text.visible = False
        self.last_update = time.time()
        if self.content_view_text.visible:
            pybullet.setCollisionFilterGroupMask(self.content_view_text.collider_id, -1, 1, 1)
        else:
            pybullet.setCollisionFilterGroupMask(self.content_view_text.collider_id, -1, 0, 0)
        if self.content_view_image.visible:
            pybullet.setCollisionFilterGroupMask(self.content_view_image.collider_id, -1, 1, 1)
        else:
            pybullet.setCollisionFilterGroupMask(self.content_view_image.collider_id, -1, 0, 0)


    def update(self, dt):
        if self.period > 0:
            t = time.time()
            if t-self.last_update > self.period:
                self.refresh_content()

    def initialize_gl(self):
        self.program = ShadersLibrary.create_program('simple_color')

    def refresh_vertices(self):
        # Refresh layout
        margin = 0.05
        h1 = self.type_label.vertices[3][1]-self.type_label.vertices[0][1]
        h2 = self.content_view_text.vertices[3][1] - self.content_view_text.vertices[0][1]
        h3 = self.content_view_image.vertices[3][1] - self.content_view_image.vertices[0][1]

        self.content_view_text.position.setY(-h2 / 2 - h1 / 2 - margin)
        self.content_view_image.position.setY(-h3 / 2 - h1 / 2 - margin)

        # Refresh vertices
        self.vertices.clear()
        fverts = list()
        fverts.append((self.content_view_text.vertices[3], self.content_view_text.position))
        fverts.append((self.type_label.vertices[0], self.type_label.position))
        fverts.append((self.content_view_text.vertices[2], self.content_view_text.position))
        fverts.append((self.type_label.vertices[1], self.type_label.position))
        fverts.append((self.content_view_image.vertices[3], self.content_view_image.position))
        fverts.append((self.type_label.vertices[0], self.type_label.position))
        fverts.append((self.content_view_image.vertices[2], self.content_view_image.position))
        fverts.append((self.type_label.vertices[1], self.type_label.position))
        for fv, p in fverts:
            self.vertices.append(QVector3D(fv[0], fv[1], fv[2])+p)

    def paint(self, program):
        self.program.bind()
        self.program.setAttributeArray(0, self.vertices)
        self.program.setUniformValue('matrix', self.proj_matrix)
        self.program.setUniformValue('material_color', QVector4D(1.0, 1.0, 1.0, 1.0))
        # TODO: Fix these lines drawing
        # GL.glDrawArrays(GL.GL_LINES, 0, int(len(self.vertices)))
        program.bind()
        return


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


class FileCodeNode(SceneNode):
    def __init__(self, name=None, parent=None, filename=None):
        if filename is None:
            raise ValueError("You have to specify a textfile")
        if name is None:
            name = "Editor: "+str(filename)
        super().__init__(name, parent=parent)
        self.filename = filename
        try:
            text = open(filename).read()
        except FileNotFoundError:
            text = " "

        self.code_editor = CodeRunnerEditorNode(name, parent=self, initial_text=text)
        self.code_editor.title_bar.set_text(filename)
        self.code_editor.text_edit.widget.textChanged.connect(self.on_text_changed)
        self.code_editor.properties["delegateGrabToParent"] = True

    def on_text_changed(self):
        f = open(self.filename, "w")
        f.write(self.code_editor.text_edit.widget.toPlainText())
        f.close()
        self.code_editor.refresh_vertices()


class CodeRunnerEditorNode(SceneNode):
    def __init__(self, name="codeRunnerEditor", parent=None, initial_text=""):
        super().__init__(name, parent=parent)

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
                                      text=initial_text,
                                      autosize=True)

        self.title_bar = TextLabelNode(name=self.name + "_titlebar", parent=self, text=name)
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

        self.vertices = list()
        self.program = None
        self.refresh_vertices()

    def initialize_gl(self):
        self.program = ShadersLibrary.create_program('simple_color')

    def refresh_vertices(self):
        # Refresh layout first
        margin = 0.05
        h1 = self.title_bar.vertices[3][1]-self.title_bar.vertices[0][1]
        h2 = self.text_edit.vertices[3][1] - self.text_edit.vertices[0][1]
        h3 = self.output_text.vertices[3][1] - self.output_text.vertices[0][1]
        self.title_bar.position.setY(h2/2 + h1/2 + margin)
        self.output_text.position.setY(-h2/2 - h3 / 2 - margin)

        # Now refresh vertices
        self.vertices.clear()
        fverts = list()
        fverts.append((self.text_edit.vertices[3], QVector3D()))
        fverts.append((self.title_bar.vertices[0], self.title_bar.position))
        fverts.append((self.text_edit.vertices[2], QVector3D()))
        fverts.append((self.title_bar.vertices[1], self.title_bar.position))
        fverts.append((self.text_edit.vertices[0], QVector3D()))
        fverts.append((self.output_text.vertices[3], self.output_text.position))
        fverts.append((self.text_edit.vertices[1], QVector3D()))
        fverts.append((self.output_text.vertices[2], self.output_text.position))
        for fv, p in fverts:
            self.vertices.append(QVector3D(fv[0], fv[1], fv[2])+p)

    def run_code(self, mode=1):
        self.output_text.set_text(" ")
        if mode == 1:
            self.redirected_output_process = StdoutQueue()
            self.process = multiprocessing.Process(target=self.threaded_func, args=(mode, self.redirected_output_process,))
            self.process.start()
        elif mode == 2:
            self.redirected_output_thread = StringIO()
            # self.thread = KillableThread(target=self.threaded_func, args=(mode, self.redirected_output_process))
            self.thread = threading.Thread(target=self.threaded_func, args=(mode, self.redirected_output_process))
            self.thread.start()

    def threaded_func(self, mode, stdout_queue):
        if mode == 1:
            sys.stdout = stdout_queue
        elif mode == 2:
            self.redirected_output_thread = stdout_helpers.redirect()
        # new_globals = globals()
        # new_locals = locals()
        # new_globals = dict()
        new_locals = dict()
        try:
            exec(self.text_edit.text, self.context.execution_context)
        except KeyboardInterrupt:
            pass

    def update(self, dt):
        if self.redirected_output_process is not None:
            try:
                newtext = self.output_text.text
                newtext += self.redirected_output_process.get(False)
                self.output_text.set_text(newtext)
                self.refresh_vertices()
            except queue.Empty:
                pass

        if self.redirected_output_thread is not None:
            if self.redirected_output_pos != self.redirected_output_thread.tell():
                self.redirected_output_pos = self.redirected_output_thread.tell()
                self.output_text.set_text(self.redirected_output_thread.getvalue())
                self.refresh_vertices()

        if (self.process is not None and self.process.is_alive()) or \
                (self.thread is not None and self.thread.is_alive()):
            self.indicator_behavior.speed = 200
            self.run_indicator.tooltip = "Running"

        else:
            self.indicator_behavior.speed = 0
            self.run_indicator.tooltip = "Not running"

        try:
            while True:
                obj, period = globals().get("_wol_render_list", []).pop()
                dv = DataViewer(parent=self.context.scene, target=obj, period=period)
                cam = self.context.current_camera
                dv.position = cam.look_at
                dir = cam.look_at - cam.position
                dv.position = cam.position + dir.normalized()*5.0
                dv.orientation = cam.orientation
                dv.orientation *= QQuaternion.fromAxisAndAngle(0, 1, 0, 180)
        except IndexError:
            pass

    def paint(self, program):
        stc = self.context.current_camera.get_behavior("SnapToCamera")
        if stc.target is self.text_edit and stc.grabbed_something:
            return
        self.program.bind()
        self.program.setAttributeArray(0, self.vertices)
        self.program.setUniformValue('matrix', self.proj_matrix)
        self.program.setUniformValue('material_color', QVector4D(1.0, 1.0, 1.0, 1.0))
        GL.glDrawArrays(GL.GL_LINES, 0, int(len(self.vertices)))
        program.bind()
        return
