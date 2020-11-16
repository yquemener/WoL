from PyQt5.QtGui import QColor, QOpenGLTexture, QImage, QWindow, QPalette
from PyQt5.QtWidgets import QLabel, QFrame, QMainWindow, QApplication, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QRect

from wol import utils
from wol.GeomNodes import CardNode


class TextLabelNode(CardNode):
    def __init__(self, parent, text="label", name="LabelNode"):
        CardNode.__init__(self, name=name, parent=parent)
        self.frame = QMainWindow()
        self.widget = QLabel()
        self.widget.setWordWrap(False)
        self.widget.setStyleSheet("""
            color: rgba(255,255,255,255); 
            background-color: rgba(0,128,0,0); 
            border: 2px solid rgba(255,255,255,255);;
            """)
        self.widget.setText(text)
        qfm = self.widget.fontMetrics()
        rect = qfm.boundingRect(QApplication.desktop().geometry(), Qt.TextWordWrap, text, 4)
        self.margin = 5
        w = rect.width() + self.margin*2
        h = rect.height() + self.margin*2
        self.frame.setGeometry(0, 0, w, h)
        wscale = w/512.0
        hscale = h/512.0
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

    def update(self, dt):
        if self.needs_refresh:
            self.texture = QOpenGLTexture(QImage(self.frame.grab()))
            self.needs_refresh = False

    def set_text(self, t):
        if t == self.text:
            return
        self.text = t
        self.widget.setText(t)
        qfm = self.widget.fontMetrics()
        rect = qfm.boundingRect(QApplication.desktop().geometry(), Qt.TextWordWrap, t, 4)
        w = rect.width() + self.margin*2
        h = rect.height() + self.margin*2
        self.frame.setGeometry(0, 0, w, h)
        self.vertices = utils.generate_square_vertices_fan()
        wscale = w/512.0
        hscale = h/512.0
        for v in self.vertices:
            v[1] *= hscale
            v[0] *= wscale
        self.refresh_vertices()
        self.needs_refresh = True


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
            self.texture = QOpenGLTexture(QImage(self.widget.grab()))
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
        wscale = w/512.0
        hscale = h/512.0
        for v in self.vertices:
            v[1] *= hscale
            v[0] *= wscale
        self.refresh_vertices()
        self.needs_refresh = True


