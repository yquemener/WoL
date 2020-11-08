from PyQt5.QtGui import QOpenGLTexture, QImage, QTextCursor, QQuaternion, QVector3D
from PyQt5.QtWidgets import QPlainTextEdit
from PyQt5.QtCore import Qt
from io import StringIO
from wol.GeomNodes import CardNode

import sys


class ConsoleNode(CardNode):
    def __init__(self, parent, name="GuiNode"):
        CardNode.__init__(self, name=name, parent=parent)
        self.widget = QPlainTextEdit()
        self.widget.setGeometry(0, 0, 512, 512)
        self.needs_refresh = True
        self.focused = False
        self.history = list()
        self.history_cursor = 0
        self.locals = locals()
        self.focusable = True

    def update(self, dt):
        if self.focused:
            self.needs_refresh = True
        if self.needs_refresh:
            self.texture = QOpenGLTexture(QImage(self.widget.grab()))
            self.needs_refresh = False

    def on_click(self, pos, evt):
        return

    def keyPressEvent(self, evt):
        if evt.key() == Qt.Key_Up:
            self.history_cursor = max(-len(self.history), self.history_cursor-1)
            s = self.widget.toPlainText()
            s = "\n".join(s.split("\n")[:-1])+"\n"+self.history[self.history_cursor]
            self.widget.setPlainText(s)
            self.widget.moveCursor(QTextCursor.End)
        elif evt.key() == Qt.Key_Down:
            if self.history_cursor != 0:
                self.history_cursor = self.history_cursor + 1
                s = self.widget.toPlainText()
                s = "\n".join(s.split("\n")[:-1]) + "\n" + self.history[self.history_cursor]
                self.widget.setPlainText(s)
                self.widget.moveCursor(QTextCursor.End)
        elif evt.key() == Qt.Key_Return:
            cmd = self.widget.toPlainText().split("\n")[-1]
            self.history.append(cmd)
            old_stdout = sys.stdout
            try:
                redirected_output = sys.stdout = StringIO()
                try:
                    a = eval(cmd, globals(), self.locals)
                    if redirected_output.getvalue()=="":
                        sys.stdout.write(repr(a)+"\n")
                except Exception:
                    exec(cmd, globals(), self.locals)
                sys.stdout = old_stdout
                result = redirected_output.getvalue()
            except Exception as e:
                result = "Exception: "+str(e)+"\n"
            finally:
                sys.stdout = old_stdout
            self.widget.moveCursor(QTextCursor.End)
            self.widget.insertPlainText("\n"+str(result))
            self.history_cursor = 0
        else:
            self.widget.keyPressEvent(evt)
        self.widget.ensureCursorVisible()

    def inputMethodEvent(self, evt):
        return self.widget.inputMethodEvent(evt)
