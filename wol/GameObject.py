from PyQt5.QtGui import QVector3D

from wol.Constants import Events
from wol.GuiElements import TextLabelNode, CodeSnippetReceiver
from wol.SceneNode import SceneNode


class GameObject(SceneNode):
    def __init__(self, parent, name="GameObject"):
        super().__init__(parent=parent, name=name)

        self.title_bar = TextLabelNode(name=self.name + "_titlebar", parent=self, text=name)
        self.title_bar.position = QVector3D(0, 1, 0)

        self.slots = list()
        self.add_slot("update")
        self.add_slot("on_click")
        self.layout()

        for c in self.children:
            c.properties["delegateGrabToParent"] = True

    def add_slot(self, name):
        slot = CodeSnippetReceiver(parent=self)
        slot.set_text("# "+name+" code\n\n\n\fdssafdas")
        slot.events_handlers.append(lambda ev: self.on_lost_focus(ev, slot))
        label = TextLabelNode(name=name + "_slot_label", parent=self, text=name)
        self.slots.append((name, label, slot))

    def on_lost_focus(self, evt, slot):
        print(evt)
        if evt == Events.LostFocus:
            self.layout()
        if evt == Events.AnimationFinished and not slot.focused:
            self.layout()

    def layout(self):
        margin = 0.03
        stc = self.context.current_camera.get_behavior("SnapToCamera")
        y = self.title_bar.position.y()
        y -= self.title_bar.hscale
        self.title_bar.position.setX(self.title_bar.wscale)
        for s in self.slots:
            y -= s[1].hscale
            s[1].position.setY(y)
            s[1].position.setX(s[1].wscale)
            y -= s[1].hscale
            y -= s[2].hscale
            s[2].position.setY(y)
            s[2].position.setX(s[2].wscale)
            y -= s[2].hscale + margin
