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
        self.add_slot("init",
"""self.quadric = GLU.gluNewQuadric()
self.size = 1.0
self.collider = Collisions.Sphere()
self.color = QVector4D(1.0, 0.5, 0.5, 1.0)
self.program = ShadersLibrary.create_program('simple_lighting')""")

        self.add_slot("update")
        self.add_slot("on_click")
        self.add_slot("paint",
"""self.program.bind()
self.program.setUniformValue('light_position', QVector3D(1.0, 10.0, -10.0))
self.program.setUniformValue('matmodel', self.transform)
self.program.setUniformValue('material_color', self.color)
self.program.setUniformValue('mvp', self.proj_matrix)
GLU.gluSphere(self.quadric, self.size, 20, 20)
program.bind()""")
        self.layout()

        for c in self.children:
            c.properties["delegateGrabToParent"] = True

    def add_slot(self, name, text="# code"):
        slot = CodeSnippetReceiver(parent=self)
        slot.set_text(text)
        slot.events_handlers[Events.LostFocus].append(lambda: self.on_lost_focus(slot))
        slot.events_handlers[Events.AnimationFinished].append(lambda: self.on_lost_focus(slot))
        label = TextLabelNode(name=name + "_slot_label", parent=self, text=name)
        self.slots.append((name, label, slot))

    def on_lost_focus(self, slot):
        if not slot.focused:
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
