import threading

from PyQt5.QtGui import QVector3D

from wol import Behavior, stdout_helpers
from wol.CodeEdit import DataViewer
from wol.Constants import Events, UserActions
from wol.GuiElements import TextLabelNode
from wol.SceneNode import SceneNode
from wol.TextEditNode import TextEditNode


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
        self.thread = None
        self.stdout = None
        self.last_pos = 0

    def on_execute(self):
        self.thread = threading.Thread(target=self.threaded_function)
        self.thread.start()

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
                print(self.last_pos, pos)
                self.obj.stdout += self.stdout.getvalue()[self.last_pos:pos+1]
                self.last_pos = pos


class NotebookNode(SceneNode):
    def __init__(self, parent, name):
        SceneNode.__init__(self, parent=parent, name=name)
        self.filename = f"my_project/{self.name}.py"
        self.cells = list()
        self.output_text = TextLabelNode(name=self.name + "_output", parent=self, text="")
        self.output_text.min_size = (400, 30)
        self.output_text.visible = False
        # self.events_handlers[Events.AppClose].append(self.save)
        # self.events_handlers[UserActions.Save].append(self.save)
        self.watcher_add_list = list()
        self.context.execution_context['watch'] = self.watcher_add_list.append

        # try:
        #     s = open(self.filename).read()
        #     sep = "\n" + "#**" * 20 + "\n"
        #     for i, celltext in enumerate(s.split(sep)):
        #         cell = self.add_cell(i)
        #         cell.set_text(celltext)
        #         cell.do_autosize()
        # except FileNotFoundError:
        #     self.add_cell(1)
        self.layout()

    def add_cell(self, num):
        cell = TextEditNode(parent=self, name=self.name + "_cell"+str(num))
        cell.add_behavior(EditOnClick())
        cell.add_behavior(ExecuteBehavior())
        cell.events_handlers[Events.TextChanged].append(self.layout)
        cell.properties["delegateGrabToParent"] = True
        cell.stdout = ""
        cell.autosize = True
        cell.min_size = (200, 30)
        cell.do_autosize()
        self.add_child(cell)
        self.cells.append(cell)
        return cell

    def layout(self):
        y = 0
        for cell in self.cells:
            y -= cell.hscale+0.01
            cell.position = QVector3D(cell.wscale, y, 0)
            # cell.do_autosize()
            y -= cell.hscale

    def update(self, dt):
        if len(self.cells) == 0 or self.cells[-1].text != "":
            self.add_cell(len(self.cells)+1)
            self.layout()
        for cell in self.cells:
            if cell.stdout != "":
                self.output_text.visible = True
                self.output_text.set_text(self.output_text.text + cell.stdout)
                cell.stdout = ""
                self.output_text.position = QVector3D(-self.output_text.wscale, -self.output_text.hscale, 0)
        for watch in self.watcher_add_list:
            DataViewer(parent=self.context.scene, target=watch, period=1)
        self.watcher_add_list.clear()

    def save(self):
        # Use notebook formats?
        f = open(self.filename, "w")
        sep = "\n" + "#**"*20 + "\n"
        f.write(sep.join([c.text for c in self.cells]))
        f.close()

    def serialize(self, current_obj_num):
        s, next_num = super().serialize(current_obj_num)
        for i, cell in enumerate(self.cells):
            s += f"cell = obj_{current_obj_num}.add_cell({i})\n"
            s += f"cell.set_text({repr(cell.text)})\n"
        s += f"cell = obj_{current_obj_num}.layout()\n"
        return s, next_num
