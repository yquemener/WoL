from wol.CodeEdit import CodeBumperNode
from wol.GuiElements import TextLabelNode
import inspect
from PyQt5.QtGui import QVector3D


class ObjectEditorNode(TextLabelNode):
    def __init__(self, parent=None, name="ObjectEditorNode", target_object=None):
        TextLabelNode.__init__(self, text=str(type(target_object)), name=name, parent=parent)
        self.children_visible = False
        self.target_object = target_object
        code = inspect.getsource(target_object.__class__)
        self.class_header_source = code.split("def __init__(self")[0].split('\n')[0]

        try:
            y = -0.15
            for member in inspect.getmembers(target_object):
                if inspect.ismethod(member[1]):
                    fn = "pieces/"
                    fn += str(type(target_object))
                    fn += "::" + str(member[0])
                    f = open(fn, "w")
                    f.write(inspect.getsource(member[1]))
                    f.close()
                    o = CodeBumperNode(parent=self, filename=fn, text=member[0])
                    o.position = QVector3D(0.3, y, 0.0)
                    y -= 0.15
            self.widget.setStyleSheet("QWidget{color: white; background-color: gray;}")
        except Exception as e:
            print(e)
            self.widget.setStyleSheet("QWidget{color: red; background-color: gray;}")
        finally:
            self.refresh_children()

    def on_click(self, pos, evt):
        self.children_visible = not self.children_visible
        self.refresh_children()

    def on_save(self, pos):
        print(self.class_header_source)
        for c in self.children:
            print(c.edit_node.widget.toPlainText())

    def refresh_children(self):
        for c in self.children:
            c.visible = self.children_visible
