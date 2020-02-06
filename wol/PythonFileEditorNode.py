from wol.CodeBumperNode import CodeBumperNode
from wol.GuiElements import TextLabelNode
from PyQt5.QtGui import QVector3D
import ast


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
                    snippet = "\n".join(self.code_lines[member.lineno-1:member.endlineno])
                    print(member.lineno, member.endlineno)
                    if hasattr(member, 'name'):
                        label = member.name
                    else:
                        label = str(type(member))
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
