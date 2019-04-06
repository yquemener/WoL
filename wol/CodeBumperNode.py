""" Object that executes a python code when clicked or space-clicked and that opens a text
editor with the code when e-clicked. Hopefully also saves the code from one session to the other.
"""
from PyQt5.QtGui import QVector3D

from wol.GuiElements import TextLabelNode
from wol.TextEditNode import TextEditNode


class CodeBumperNode(TextLabelNode):
    def __init__(self, parent=None, name="CodeBumber", text="BUMP", filename="pieces/yq_1"):
        TextLabelNode.__init__(self, text=text, name=name, parent=parent)
        self.edit_node = TextEditNode(parent=self, name=self.name+"#edit", filename=filename)
        self.edit_node.visible = False
        self.edit_node.position = QVector3D(1.2, 0, 0)
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
