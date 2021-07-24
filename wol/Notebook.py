import threading

from OpenGL import GL
from PyQt5 import QtCore
from PyQt5.QtGui import QVector3D, QVector4D
from PyQt5.QtWidgets import QApplication

from wol import Behavior, stdout_helpers
from wol.Behavior import RotateConstantSpeed
from wol.CodeEdit import DataViewer
from wol.Constants import Events, UserActions
from wol.GeomNodes import CubeNode
from wol.GuiElements import TextLabelNode
from wol.SceneNode import SceneNode
from wol.ShadersLibrary import ShadersLibrary
from wol.TextEditNode import TextEditNode


class ExecuteBehavior(Behavior.Behavior):
    def __init__(self):
        super().__init__()
        self.events_handlers[UserActions.Execute].append(self.on_execute)
        self.thread = None
        self.stdout = None
        self.last_pos = 0

    def on_execute(self):
        self.thread = threading.Thread(target=self.threaded_function)
        self.thread.start()
        self.obj.run_indicator.visible = True
        rb = self.obj.run_indicator.get_behavior("RotateConstantSpeed")
        rb.reset()
        rb.speed = 100

    def threaded_function(self):
        try:
            self.stdout = stdout_helpers.redirect()
            stdout_helpers.enable_proxy()
            self.obj.stdout = ""
            self.last_pos = 0
            exec(self.obj.text, self.obj.context.execution_context)
        except Exception as e:
            print(f"Cell execution failed with: {e}")

    def on_update(self, dt):
        if self.stdout is not None:
            pos = self.stdout.tell()
            if pos != self.last_pos:
                self.obj.stdout += self.stdout.getvalue()[self.last_pos:pos+1]
                self.last_pos = pos
        if self.thread is None or not self.thread.is_alive():
            self.obj.run_indicator.visible = False


class NotebookNode(SceneNode):
    def __init__(self, parent, name):
        SceneNode.__init__(self, parent=parent, name=name)
        self.cells = list()
        self.title_label = TextLabelNode(parent=self, text=str(self.name))
        self.title_label.position = QVector3D(self.title_label.wscale, self.title_label.hscale, 0)
        self.output_text = TextLabelNode(name=self.name + "_output", parent=self, text="")

        self.button_close = CubeNode(parent=self, color=QVector4D(0.8, 0.2, 0.2, 0.8))
        self.button_close.position = QVector3D(self.title_label.wscale*2+0.1, self.title_label.hscale, 0)
        self.button_close.scale = QVector3D(0.07, 0.07, 0.07)
        self.button_close.events_handlers[Events.Clicked].append(lambda:  self.remove())
        self.button_close.tooltip = "Close"

        self.button_close = CubeNode(parent=self, color=QVector4D(0.6, 0.6, 0.6, 0.8))
        self.button_close.position = QVector3D(self.title_label.wscale*2+0.3, self.title_label.hscale, 0)
        self.button_close.scale = QVector3D(0.07, 0.07, 0.07)
        self.button_close.events_handlers[Events.Clicked].append(lambda:  self.output_text.set_text(""))
        self.button_close.tooltip = "Clear stdout"

        self.output_text.min_size = (400, 30)
        self.output_text.visible = False
        self.output_text.properties["delegateGrabToParent"] = True

        self.watcher_add_list = list()
        self.context.execution_context['watch'] = lambda v, dur=1.0: self.watcher_add_list.append((v, dur))
        self.events_handlers[UserActions.Unselect].append(lambda: self.select_cell(None))
        self.events_handlers[UserActions.Edit].append(self.on_start_edit_cell)
        self.edit_cell_mode = False
        self.focused = False
        self.selected_cell = None
        self.vertices = list()
        self.program = None
        self.layout()

    def select_cell(self, new_cell):
        if self.selected_cell == new_cell is None:
            return
        if self.selected_cell is not None:
            self.cell_border(self.selected_cell)
            self.selected_cell.focused = False

        self.selected_cell = new_cell
        if self.selected_cell is not None:
            self.cell_border(self.selected_cell, (255,255,0,255))
            self.selected_cell.focused = True

    def cell_border(self, cell, color=None):
        if color is None:
            cell.widget.setStyleSheet("QWidget{color: white; background-color: black;}")
        else:
            cell.widget.setStyleSheet(f"""
                        color: rgba(255,255,255,255);
                        background-color: rgba(0,0,0,255);
                        border: 2px solid rgba{color};;
                        """)
        cell.needs_refresh = True

    def keyPressEvent(self, evt):
        if evt.key() == QtCore.Qt.Key_Return:
            if QApplication.keyboardModifiers() == QtCore.Qt.ShiftModifier:
                if self.selected_cell is not None:
                    self.selected_cell.on_event(UserActions.Execute)
                    self.edit_cell_mode = False
                    ind = self.cells.index(self.selected_cell)
                    ind = max(0, min(ind, len(self.cells) - 2))
                    self.select_cell(self.cells[ind + 1])
            else:
                if self.edit_cell_mode:
                    self.selected_cell.keyPressEvent(evt)
                    self.selected_cell.needs_refresh = True
                    self.selected_cell.do_autosize()
                    self.layout()
                else:
                    self.on_event(UserActions.Edit)
        elif evt.key() == QtCore.Qt.Key_Escape:
            if self.edit_cell_mode:
                self.edit_cell_mode = False
                self.cell_border(self.selected_cell, (255,255,0,255))
            else:
                self.select_cell(None)
                self.context.focus(None)

        elif evt.key() == QtCore.Qt.Key_Down and not self.edit_cell_mode:
            ind = self.cells.index(self.selected_cell)
            ind = max(0, min(ind, len(self.cells)-2))
            self.select_cell(self.cells[ind+1])

        elif evt.key() == QtCore.Qt.Key_Up and not self.edit_cell_mode:
            ind = self.cells.index(self.selected_cell)
            ind = max(1, min(ind, len(self.cells) - 1))
            self.select_cell(self.cells[ind - 1])

        else:
            if self.selected_cell is not None and self.edit_cell_mode:
                self.selected_cell.keyPressEvent(evt)
                self.selected_cell.needs_refresh = True
                self.selected_cell.do_autosize()
                self.layout()

    def on_click_cell(self, cell):
        self.select_cell(cell)
        self.context.focus(self)

    def on_start_edit_cell(self):
        if self.selected_cell is not None:
            self.edit_cell_mode = True
            self.cell_border(self.selected_cell, (0, 0, 255, 255))

    def on_finish_edit_cell(self, cell):
        self.select_cell(cell)
        self.edit_cell_mode = False

    def add_cell(self, num):
        cell = TextEditNode(parent=self, name=self.name + "_cell"+str(num))
        cell.add_behavior(ExecuteBehavior())
        cell.events_handlers[Events.TextChanged].append(self.layout)
        cell.events_handlers[Events.Clicked].append(
            lambda: self.on_click_cell(cell))
        cell.events_handlers[UserActions.Unselect].append(lambda: self.on_finish_edit_cell(cell))
        cell.properties["delegateGrabToParent"] = True
        cell.stdout = ""
        cell.autosize = True
        cell.min_size = (200, 30)
        cell.do_autosize()
        cell.run_indicator = CubeNode(parent=cell, name=cell.name+"_run_indicator")
        cell.run_indicator.visible = False
        cell.run_indicator.add_behavior(RotateConstantSpeed(0))
        cell.run_indicator.scale = QVector3D(0.1, 0.1, 0.1)
        cell.add_child(cell.run_indicator)
        self.add_child(cell)
        self.cells.append(cell)
        return cell

    def layout(self):
        y = 0
        self.vertices.clear()
        self.vertices.append(QVector3D(0, 0, -0.5))
        for cell in self.cells:
            y -= cell.hscale+0.01
            cell.position = QVector3D(cell.wscale, y, 0)
            self.vertices.append(QVector3D(0, y, -0.5))
            self.vertices.append(QVector3D(0, y, -0.5))
            self.vertices.append(QVector3D(cell.wscale, y, 0.0))
            self.vertices.append(QVector3D(0, y, -0.5))
            cell.run_indicator.position = QVector3D(cell.wscale+0.15, 0, 0)
            # cell.do_autosize()
            y -= cell.hscale

    def update(self, dt):
        # Adds an empty cell at the end
        if len(self.cells) == 0 or self.cells[-1].text != "":
            self.add_cell(len(self.cells)+1)
            self.layout()

        # Gather the stdout executed by the cells' threads
        cell_focused = False
        for cell in self.cells:
            if cell.stdout != "":
                self.output_text.visible = True
                self.output_text.set_text(self.output_text.text + cell.stdout)
                cell.stdout = ""
                self.output_text.position = QVector3D(-self.output_text.wscale, -self.output_text.hscale, 0)
            cell_focused = cell.focused | cell_focused

        # Deferred watcher creation because PyQt refuses that different widgets live in
        # different threads
        for (watch, duration) in self.watcher_add_list:
            DataViewer(parent=self.context.scene, target=watch, period=duration)
        self.watcher_add_list.clear()

    def serialize(self, current_obj_num):
        s, next_num = super().serialize(current_obj_num)
        for i, cell in enumerate(self.cells):
            s += f"cell = obj_{current_obj_num}.add_cell({i})\n"
            s += f"cell.set_text({repr(cell.text)})\n"
        s += f"cell = obj_{current_obj_num}.layout()\n"
        return s, next_num

    def initialize_gl(self):
        self.program = ShadersLibrary.create_program('simple_color')

    def paint(self, program):
        self.program.bind()
        self.program.setAttributeArray(0, self.vertices)
        self.program.setUniformValue('matrix', self.proj_matrix)
        self.program.setUniformValue('material_color', QVector4D(1.0, 1.0, 1.0, 1.0))
        GL.glDrawArrays(GL.GL_LINES, 0, int(len(self.vertices)))
        program.bind()
