from PyQt5.QtGui import QColor, QOpenGLTexture, QImage
from PyQt5.QtWidgets import QLabel

from wol.GeomNodes import CardNode


class TextLabelNode(CardNode):
    def __init__(self, parent, text="label", name="LabelNode"):
        CardNode.__init__(self, name=name, parent=parent)
        self.widget = QLabel()
        self.widget.setGeometry(0, 0, 512, 512)
        self.widget.setText(text)
        qfm = self.widget.fontMetrics()
        w = qfm.width(text)
        h = qfm.height()
        self.widget.setGeometry(0, 0, w, h)
        wscale = w/512.0
        hscale = h/512.0
        for v in self.vertices:
            v[1] *= hscale
            v[0] *= wscale
        self.refresh_vertices()
        self.needs_refresh = True
        self.text = text
        self.widget.setStyleSheet("QWidget{color: white; background-color: gray;}");

    def update(self, dt):
        if self.needs_refresh:
            self.texture = QOpenGLTexture(QImage(self.widget.grab()))
            self.needs_refresh = False

