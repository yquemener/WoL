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
    def __init__(self, parent, target, name="GameObject"):
        super().__init__(parent=parent, name=name)

        self.target = target
        self.title_bar = TextLabelNode(name=self.name + "_titlebar", parent=self, text=str(target.__class__))
        self.title_bar.position = QVector3D(0, 1, 0)
        self.slots = list()

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
                self.add_slot(last_name, current_code.rstrip(' \t\n'))
                current_code = ""
                indent, last_name, args = m.group(1), m.group(2), m.group(3)
            current_code += line[len(indent):]+"\n"
        self.add_slot(name, current_code.rstrip(' \t\n'))
        self.add_slot("next", " ")

        self.layout()

        for c in self.children:
            c.properties["delegateGrabToParent"] = True

    def add_slot(self, name, text="# code"):
        slot = CodeSnippetReceiver(parent=self)
        slot.set_text(text)
        slot.events_handlers[Events.LostFocus].append(lambda: self.on_lost_focus(slot))
        slot.events_handlers[Events.AnimationFinished].append(lambda: self.on_lost_focus(slot))
        slot.events_handlers[UserActions.Unselect].append(lambda: self.on_code_update(slot))
        # label = TextLabelNode(name=name + "_slot_label", parent=self, text=name)
        label = None
        self.slots.append((name, label, slot))

    def on_lost_focus(self, slot):
        if not slot.focused:
            self.layout()

    def on_code_update(self, slot):
        name, func = compile_function(slot.text)
        setattr(self.target.__class__, name, func)
        code = self.slots[0][2].text
        for _, _, slot in self.slots[1:]:
            for line in slot.text.split("\n"):
                code += self.context.indent + line + "\n"
            code += "\n"
        self.target.code = code

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

