import time
from PyQt5.QtGui import QColor, QOpenGLTexture, QImage, QWindow, QPalette
from PyQt5.QtWidgets import QLabel, QFrame, QMainWindow, QApplication, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QRect

from wol import utils
from wol.Behavior import Behavior
from wol.Constants import UserActions
from wol.GeomNodes import CardNode
from wol.TextEditNode import TextEditNode


class TextLabelNode(CardNode):
    def __init__(self, parent, text="label", name="LabelNode"):
        CardNode.__init__(self, name=name, parent=parent)
        self.frame = QMainWindow()
        self.frame.setStyleSheet("""
            color: rgba(255,255,255,255); 
            background-color: rgba(0,128,0,100); 
            border: 2px solid rgba(255,255,255,255);;
            """)

        self.widget = QLabel()
        self.widget.setWordWrap(False)
        self.widget.setStyleSheet("""
            color: rgba(255,255,255,255); 
            background-color: rgba(0,128,0,0); 
            border: 0px solid rgba(255,255,255,255);;
            """)
        self.widget.setText(text)
        qfm = self.widget.fontMetrics()
        rect = qfm.boundingRect(QApplication.desktop().geometry(), Qt.TextWordWrap, text, 4)
        self.margin = 15
        w = rect.width() + self.margin * 2
        h = rect.height() + self.margin * 2
        self.frame.setGeometry(0, 0, w, h)
        wscale = w / 512.0
        hscale = h / 512.0
        for v in self.vertices:
            v[1] *= hscale
            v[0] *= wscale
        self.refresh_vertices()
        self.needs_refresh = True
        self.text = text
        layout = QVBoxLayout()
        layout.addWidget(self.widget)
        widget = QWidget()
        widget.setLayout(layout)
        self.frame.setCentralWidget(widget)
        self.frame.setAttribute(Qt.WA_TranslucentBackground, True)
        self.layer = 2
        self.focusable = True

    def update(self, dt):
        if self.needs_refresh:
            if self.text == "":
                self.texture = QOpenGLTexture(QImage(b'\0\0\0\0', 1, 1, QImage.Format_ARGB32))
            else:
                self.texture = QOpenGLTexture(QImage(self.frame.grab()))
            self.needs_refresh = False

    def set_text(self, t):
        if t == self.text:
            return
        self.text = t
        self.widget.setText(t)
        qfm = self.widget.fontMetrics()
        rect = qfm.boundingRect(QApplication.desktop().geometry(), Qt.TextWordWrap, t, 4)
        w = rect.width() + self.margin * 2
        h = rect.height() + self.margin * 2
        self.frame.setFixedSize(w, h)

        self.vertices = utils.generate_square_vertices_fan()
        wscale = w / 512.0
        hscale = h / 512.0
        for v in self.vertices:
            v[1] *= hscale
            v[0] *= wscale
        self.refresh_vertices()
        self.needs_refresh = True


class CodeSnippet(TextEditNode):
    def __init__(self, parent):
        super().__init__(parent=parent, autosize=True)
        self.add_behavior(CodeSnippetBehaviorPaste())
        self.add_behavior(CodeSnippetBehaviorCopy())
        self.add_behavior(CodeSnippetBehaviorCut())


class CodeSnippetReceiver(TextEditNode):
    def __init__(self, parent):
        super().__init__(parent=parent, autosize=True)
        self.widget.setStyleSheet("""
                color: rgba(255,255,255,255);
                background-color: rgba(0,0,0,100);
                border: 2px solid rgba(255,255,255,255);;
            """)
        self.set_text("Receiver")
        self.old_text = self.text
        self.hovered = False
        self.add_behavior(CodeSnippetBehaviorCopy())
        self.add_behavior(CodeSnippetBehaviorCut())

    def update(self, dt):
        if self.context.hover_target and self.context.hover_target is self\
                and not self.hovered:
            if self.context.grabbed is not None and isinstance(self.context.grabbed, CodeSnippet):
                self.widget.setStyleSheet("""
                    color: rgba(255,255,255,255); 
                    background-color: rgba(0,0,0,100); 
                    border: 10px solid rgba(0,0,200,200);;
                    """)
                self.needs_refresh = True

        if self.hovered and  \
                (not self.context.hover_target or self.context.hover_target is not self):
            self.widget.setStyleSheet("""
                color: rgba(255,255,255,255);
                background-color: rgba(0,0,0,100);
                border: 2px solid rgba(255,255,255,255);;
                """)
            self.needs_refresh = True
        super().update(dt)
        self.hovered = (self.context.hover_target is self)


class CodeSnippetBehaviorCopy(Behavior):
    def init_handlers(self):
        self.handlers[UserActions.Copy] = [self.copy]

    def copy(self):
        if self.obj.context.grabbed:
            return
        if self.obj is self.obj.context.hover_target:
            new_snippet = CodeSnippet(parent=self.obj.context.scene)
            new_snippet.position = self.obj.position
            new_snippet.orientation = self.obj.orientation
            new_snippet.compute_transform()
            new_snippet.set_text(self.obj.text)
            self.grab(new_snippet)
            # create a CodeSnippet at the same place as the target , grab it

    def grab(self, target):
        gr = self.obj.context.grabbed
        if gr is None:
            self.obj.context.grabbed = target
            self.obj.context.grabbed_former_parent = self.obj.context.grabbed.parent
            self.obj.context.grabbed.reparent(self.obj.context.current_camera)


class CodeSnippetBehaviorCut(Behavior):
    def init_handlers(self):
        self.handlers[UserActions.Cut] = [self.cut]

    def cut(self):
        if self.obj.context.grabbed:
            return
        if self.obj.context.hover_target is self.obj:
            if isinstance(self.obj, CodeSnippet):
                self.grab(self.obj)
            elif isinstance(self.obj, CodeSnippetReceiver):
                new_snippet = CodeSnippet(parent=self.obj.context.scene)
                new_snippet.position = self.obj.position
                new_snippet.orientation = self.obj.orientation
                new_snippet.compute_transform()
                new_snippet.set_text(self.obj.text)
                self.grab(new_snippet)
                self.obj.set_text(" ")

    def grab(self, target):
        gr = self.obj.context.grabbed
        if gr is None:
            self.obj.context.grabbed = target
            self.obj.context.grabbed_former_parent = self.obj.context.grabbed.parent
            self.obj.context.grabbed.reparent(self.obj.context.current_camera)


class CodeSnippetBehaviorPaste(Behavior):
    def init_handlers(self):
        self.handlers[UserActions.Paste] = [self.paste]

    def paste(self):
        target = self.obj.context.hover_target
        if target and isinstance(target, CodeSnippetReceiver):
            target.set_text(self.obj.text)
            print(self.obj.text)
            self.obj.remove()
            self.obj.context.grabbed = None
        else:
            self.ungrab()

    def ungrab(self):
        gr = self.obj.context.grabbed
        if gr is not None:
            self.obj.context.grabbed.reparent(self.obj.context.grabbed_former_parent)
            self.obj.context.grabbed = None


class WidgetTestNode(CardNode):
    def make_label(self, text):
        label = QLabel()
        # label.setGeometry(0, 0, 512, 512)
        label.setWordWrap(True)
        label.setText(text)
        label.setStyleSheet("""
            color: rgba(255,255,255,255); 
            background-color: rgba(0,128,0,0); 
            border: 2px solid rgba(255,255,255,255);;
            """)
        # qfm = label.fontMetrics()
        # rect = qfm.boundingRect(QApplication.desktop().geometry(), Qt.TextWordWrap, text, 4)
        # w = rect.width()
        # h = rect.height()
        # label.setGeometry(0, 0, w+15, h+15)
        return label

    def __init__(self, parent, text="label", name="LabelNode"):
        CardNode.__init__(self, name=name, parent=parent)

        frame = QMainWindow()
        frame.setGeometry(0, 0, 512, 512)
        layout = QVBoxLayout()
        layout.addWidget(self.make_label(text))
        layout.addWidget(self.make_label("La tête à Toto"))

        widget = QWidget()
        widget.setLayout(layout)
        frame.setCentralWidget(widget)

        #
        #
        # frame = QMainWindow()
        # widg = QWidget()
        # label = self.make_label(text)
        # label2 = self.make_label("La tête à Toto")
        # layout = QVBoxLayout()
        # layout.addChildWidget(label)
        # layout.addChildWidget(label2)
        # widg.setLayout(layout)
        # frame.setCentralWidget(widg)

        wscale = 1
        hscale = 1
        for v in self.vertices:
            v[1] *= hscale
            v[0] *= wscale
        self.refresh_vertices()
        self.needs_refresh = True
        self.text = text
        frame.setAttribute(Qt.WA_TranslucentBackground, True)
        self.widget = frame
        self.layer = 2

    def update(self, dt):
        if self.needs_refresh:
            if self.text == "":
                self.texture = QOpenGLTexture(QImage(self.frame.grab()))
            else:
                self.texture = None
            self.needs_refresh = False

    def set_text(self, t):
        if t == self.text:
            return
        self.text = t
        self.widget.setText(t)
        qfm = self.widget.fontMetrics()
        rect = qfm.boundingRect(QApplication.desktop().geometry(), Qt.TextWordWrap, t, 4)
        w = rect.width()
        h = rect.height()
        self.widget.setGeometry(0, 0, w, h)
        self.vertices = utils.generate_square_vertices_fan()
        wscale = w / 512.0
        hscale = h / 512.0
        for v in self.vertices:
            v[1] *= hscale
            v[0] *= wscale
        self.refresh_vertices()

        self.needs_refresh = True
