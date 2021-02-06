from PyQt5.QtGui import QVector3D

from wol.GuiElements import TextLabelNode, CodeSnippetReceiver
from wol.SceneNode import SceneNode


class GameObject(SceneNode):
    def __init__(self, parent, name="GameObject"):
        super().__init__(parent=parent, name=name)

        self.title_bar = TextLabelNode(name=self.name + "_titlebar", parent=self, text=name)
        self.title_bar.position = QVector3D(0, 1, 0)

        self.slot1 = CodeSnippetReceiver(parent=self)
        self.slot1.set_text("# update code")

        for c in self.children:
            c.properties["delegateGrabToParent"] = True

    def update(self, dt):
        if not self.slot1.focused:
            try:
                exec(self.slot1.text)
            except:
                print("Exception")