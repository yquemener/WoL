from wol import Behavior
from wol.CodeBumperNode import CodeBumperNode
from wol.GeomNodes import CubeNode, WireframeCubeNode
from wol.GuiElements import TextLabelNode
from PyQt5.QtGui import QVector3D, QVector4D
import ast

from wol.SceneNode import SceneNode


def recurs_find_end_line(element):
    if hasattr(element, 'lineno'):
        endlineno = element.lineno
    else:
        endlineno = 1
    if hasattr(element, '_fields') and len(element._fields)>0:
        for f in element._fields:
            child = getattr(element, f)
            if hasattr(child, "__len__") and len(child)>0:
                le = recurs_find_end_line(child[-1])
            else:
                le = recurs_find_end_line(child)
            if le>endlineno:
                endlineno = le
    return endlineno


def add_line_end_no(ast_tree):
    if hasattr(ast_tree, "body"):
        for b in ast_tree.body:
            b.endlineno = recurs_find_end_line(b)
            add_line_end_no(b)


class InstanciationBehavior(Behavior.Behavior):
    def __init__(self, code_to_execute, obj_to_instantiate):
        super().__init__()
        self.code_to_execute = code_to_execute
        self.obj_to_instantiate = obj_to_instantiate

    def on_click(self, evt, pos):
        # Execute the class statements (Insecure!)
        exec(self.code_to_execute, globals())
        # Actual instantiation:
        o = globals()[self.obj_to_instantiate](10,10)
        obj = InstanciatedObject(obj=o, parent=self.obj.context.scene)
        obj.position = self.obj.world_position() + QVector3D(1, 0, 0)


class PythonFileEditorNode(TextLabelNode):
    def __init__(self, parent=None, name="PythonFileEditorNode", target_file_name=None):
        TextLabelNode.__init__(self, text=target_file_name, name=name, parent=parent)
        self.children_visible = False
        self.full_code = open(target_file_name, "r").read()
        self.code_lines = self.full_code.split('\n')
        root = ast.parse(self.full_code)
        add_line_end_no(root)

        try:
            y = -0.15
            for member in root.body:
                if hasattr(member, 'lineno') and hasattr(member, 'endlineno'):
                    # snippet = "\n".join(self.code_lines[member.lineno-1:member.endlineno])
                    snippet = self.full_code
                    if hasattr(member, 'name'):
                        label = member.name
                    else:
                        label = str(type(member))
                    if type(member) is ast.ClassDef:
                        wc = WireframeCubeNode(parent=self, color=QVector4D(1, 1, 1, 1))
                        wc.add_behavior(Behavior.RotateConstantSpeed(80.0))
                        wc.scale = QVector3D(0.05, 0.05, 0.05)
                        wc.position = QVector3D(-0.3, y, 0.0)
                        wc.add_behavior(InstanciationBehavior(snippet, member.name))
                    o = CodeBumperNode(parent=self,
                                       filename=None,
                                       label=label,
                                       code=snippet)
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


# What is the goal of that object?
# - Show the members and their states
#  -> object explorer
# - Call member functions
#  -> parameter editor

class InstanciatedObject(SceneNode):
    def __init__(self, obj, name=None, parent=None):
        SceneNode.__init__(self, name=name, parent=parent)
        if name is None:
            name = obj.__class__.__name__
        self.name = name
        self.instanciated_object = obj
        self.icon = CubeNode(parent=self)
        self.icon.properties["delegateGrabToParent"] = True
        self.icon.add_behavior(Behavior.RotateConstantSpeed())
        self.icon.scale = QVector3D(0.05, 0.05, 0.05)
        self.icon.position = QVector3D(-0.2, 0.0, 0.0)
        self.icon.add_behavior(Behavior.TransmitClickToParent())

        self.label = TextLabelNode(parent=self, text=self.instanciated_object.__class__.__name__)
        self.label.properties["delegateGrabToParent"] = True
        self.label.add_behavior(Behavior.TransmitClickToParent())

        self.members_visible = False
        self.member_objects = list()

        y = -0.10
        for member in dir(self.instanciated_object):
            if member.startswith("__"):
                continue
            label = TextLabelNode(parent=self, text=member)
            label.position = QVector3D(0.3, y, 0)
            self.member_objects.append(label)
            y -= 0.10
        self.refresh_members()

    def refresh_members(self):
        for c in self.member_objects:
            c.visible = self.members_visible

    def on_click(self, pos, evt):
        self.members_visible = not self.members_visible
        self.refresh_members()
