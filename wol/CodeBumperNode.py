""" Object that executes a python code when clicked or space-clicked and that opens a text
editor with the code when e-clicked. Hopefully also saves the code from one session to the other.
"""
from PyQt5.QtGui import QVector3D

from wol.GeomNodes import CardNode
from wol.TextEditNode import TextEditNode


class CodeBumperNode(CardNode):
    def __init__(self, parent=None, name="CodeBumber"):
        CardNode.__init__(self, filename="resources/sample.jpg", name=name, parent=parent)
        self.edit_node = TextEditNode(parent=self, name=self.name+"#edit")
        self.edit_node.visible = False
        self.edit_node.position = QVector3D(0,0,-1)

    def on_click(self, pos, evt):
        print("BUMP")
        self.edit_node.visible = not self.edit_node.visible
