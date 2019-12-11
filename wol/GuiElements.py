from PyQt5.QtGui import QColor, QOpenGLTexture, QImage, QWindow
from PyQt5.QtWidgets import QLabel, QFrame, QMainWindow, QApplication
from PyQt5.QtCore import Qt, QRect

from wol import utils
from wol.GeomNodes import CardNode


class TextLabelNode(CardNode):
    def __init__(self, parent, text="label", name="LabelNode"):
        CardNode.__init__(self, name=name, parent=parent)
        self.widget = QLabel()
        self.widget.setGeometry(0, 0, 512, 512)
        self.widget.setWordWrap(True)
        self.widget.setText(text)
        qfm = self.widget.fontMetrics()
        rect = qfm.boundingRect(QApplication.desktop().geometry(), Qt.TextWordWrap, text, 4)
        w = rect.width()
        h = rect.height()
        self.widget.setGeometry(0, 0, w, h)
        wscale = w/512.0
        hscale = h/512.0
        for v in self.vertices:
            v[1] *= hscale
            v[0] *= wscale
        self.refresh_vertices()
        self.needs_refresh = True
        self.text = text
        self.widget.setAttribute(Qt.WA_TranslucentBackground, True)
        self.widget.setStyleSheet("color: rgba(255,255,255,255); background-color: rgba(128,0,0,255);");
        self.layer = 2

    def update(self, dt):
        if self.needs_refresh:
            self.texture = QOpenGLTexture(QImage(self.widget.grab()))
            self.needs_refresh = False

    def set_text(self, t):
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


