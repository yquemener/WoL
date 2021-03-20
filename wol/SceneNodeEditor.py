from PyQt5.QtGui import QVector3D

from wol.Behavior import Behavior
from wol.Constants import Events, UserActions
from wol.GuiElements import TextLabelNode, CodeSnippetReceiver
from wol.SceneNode import SceneNode
import re
import inspect


def compile_function(code):
    to_run = ""
    method_def_re = re.compile(r'([ \t]*)def[ \t]+([^\(]+)\((.*)\)')
    lines = code.split("\n")
    for i, l in enumerate(lines):
        m = method_def_re.match(l)
        if m is not None:
            indent, name, args = m.group(1), m.group(2), m.group(3)
            to_run = f"def {name}({args}):\n"
            to_run += "\n".join(lines[i+1:])
            print(to_run)
            exec(to_run)
            func = locals()[name]
            return name, func


class SceneNodeEditor(SceneNode):
    def __init__(self, parent, target, name="SceneNodeEditor"):
        super().__init__(parent=parent, name=name)

        self.last_recreate_failed = False
        self.target = target
        self.title_bar = TextLabelNode(name=self.name + "_titlebar", parent=self, text=str(target.__class__))
        self.title_bar.position = QVector3D(0, 1, 0)
        self.slots = list()

        if hasattr(target, "code") and target.code is not None:
            # Split the code into slots
            method_def_re = re.compile(r'([ \t]*)def[ \t]+([^\(]+)\((.*)\)')
            # whitespace_re = re.compile(r'^([ \t]*)$')
            lines = self.target.code.split("\n")
            current_code = ""
            last_name = ""
            indent = ""
            for i, line in enumerate(lines):
                line = line.rstrip('\n')
                m = method_def_re.match(line)
                if m is not None:
                    if current_code != "":
                        self.add_slot(last_name, current_code.rstrip(' \t\n'), "function")
                    indent, last_name, args = m.group(1), m.group(2), m.group(3)
                    current_code = line[len(indent):] + "\n"
                elif line.lstrip(' \t\n').startswith("class "):
                    self.add_slot(last_name, current_code.rstrip(' \t\n'), "imports")
                    self.add_slot(last_name, line.strip(' \t\n'), "class")
                    current_code = ""
                else:
                    current_code += line[len(indent):]+"\n"
            self.add_slot(name, current_code.rstrip(' \t\n'), "function")
            self.add_slot("next", " ", "function")

        self.layout()

        for c in self.children:
            c.properties["delegateGrabToParent"] = True
        self.properties["delegateGrabToParent"] = False

    def add_slot(self, name, text, slot_type):
        slot = CodeSnippetReceiver(parent=self)
        slot.set_text(text)
        slot.events_handlers[Events.LostFocus].append(lambda: self.on_lost_focus(slot))
        slot.events_handlers[Events.AnimationFinished].append(lambda: self.on_lost_focus(slot))
        slot.events_handlers[UserActions.Unselect].append(lambda: self.on_code_update(slot, slot_type))
        label = None
        self.slots.append((name, label, slot, slot_type))

    def on_lost_focus(self, slot):
        if not slot.focused:
            self.layout()

    def find_class_name(self):
        for line in self.target.code.split("\n"):
            if line.strip().startswith("class "):
                return line.strip()[6:].strip().split(":")[0].split("(")[0]
        raise SyntaxError("Could not find a class name")

    def recreate_target(self):
        self.refresh_target_code()
        classname = self.find_class_name()
        try:
            exec(self.target.code, self.context.execution_context)
            classobj = self.context.execution_context[classname]
            new_target = classobj(parent=self.target.parent)
            new_target.code = self.target.code
            new_target.position = self.target.position
            new_target.orientation = self.target.orientation
            new_target.initialize_gl()
            self.target.hide_error()
            self.target.remove()
            self.target = new_target
        except Exception as e:
            self.target.show_error(e)
            self.last_recreate_failed = True

    def refresh_target_code(self):
        code = self.slots[0][2].text + "\n\n"
        code += self.slots[1][2].text + "\n"
        for _, _, slot, slot_type in self.slots[2:]:
            for line in slot.text.split("\n"):
                code += self.context.indent + line + "\n"
            code += "\n"
        self.target.code = code

    def on_code_update(self, slot, slot_type):
        try:
            if slot_type == "function":
                name, func = compile_function(slot.text)
                setattr(self.target.__class__, name, func)
                self.refresh_target_code()
            if slot_type == "class":
                self.recreate_target()
            if slot_type == "imports":
                exec(slot.text, self.context.execution_context)
                self.refresh_target_code()
        except Exception as e:
            self.target.show_error(e)
        else:
            self.target.hide_error()

        # Create/remove blank slot at the end
        if slot is self.slots[-1][2]:
            if slot.text != "":
                self.add_slot(f"slot_{len(self.slots)}", "")
        if len(self.slots)>1 and self.slots[-2][2].text == "" and self.slots[-1][2].text == "":
            self.slots.pop(-1)[2].remove()

    def layout(self):
        margin = 0.03
        y = 0
        y += self.title_bar.position.y()
        y -= self.title_bar.hscale
        self.title_bar.position.setX(self.title_bar.wscale)
        for s in self.slots:
            # y -= s[1].hscale
            # s[1].position.setY(y)
            # s[1].position.setX(s[1].wscale)
            # y -= s[1].hscale
            y -= s[2].hscale
            s[2].position.setY(y)
            s[2].position.setX(s[2].wscale)
            y -= s[2].hscale + margin

