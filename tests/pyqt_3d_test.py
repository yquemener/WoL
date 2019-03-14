#!/usr/bin/python3
# -*- coding: utf-8 -*-


import sys
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication
from PyQt5.QtGui import QPixmap, QImage, QMouseEvent
from PyQt5.QtCore import QEvent, Qt, QPoint

from panda3d.core import Texture, CardMaker, TextureStage
from direct.showbase.ShowBase import ShowBase


class PandaApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        cm = CardMaker('card')
        self.card = self.render.attachNewNode(cm.generate())
        self.card.setPos(0.0, 3.5, 0.0)
        self.card.setScale(0.8, 0.8, 0.8)
        self.card.setTexScale(TextureStage.getDefault(), 1, -1)
        self.tex = Texture()
        self.tex.setup2dTexture(250, 250, Texture.T_unsigned_byte, Texture.F_rgba)


class Example(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def debugClick(self, *args, **kwargs):
        print(args)

    def initUI(self):
        qbtn = QPushButton('Quit', self)
        print(type(qbtn.clicked))
        qbtn.clicked.connect(self.debugClick)
        qbtn.resize(qbtn.sizeHint())
        qbtn.move(50, 50)
        self.button = qbtn

        self.setGeometry(300, 300, 250, 250)
        self.setWindowTitle('Quit button')
        self.show()

    def mousePressEvent(self, evt):
        print(evt.pos())


def qt_update(task):
    global app, ex, papp, count
    app.processEvents()
    i=QImage(ex.grab())
    buf = i.bits().asstring(250*250*4)
    papp.tex.setRamImage(buf)
    papp.card.setTexture(papp.tex)
    count+=1
    if count%100 == 0:
        e = QMouseEvent(QMouseEvent.MouseButtonPress,
                        QPoint(10,10),
                        Qt.LeftButton,
                        Qt.LeftButton,
                        Qt.NoModifier)
        QApplication.instance().sendEvent(ex.button, e)
        print(count)
    if count%100 == 15:
        e = QMouseEvent(QMouseEvent.MouseButtonRelease,
                        QPoint(10,10),
                        Qt.LeftButton,
                        Qt.LeftButton,
                        Qt.NoModifier)
        QApplication.instance().sendEvent(ex.button, e)
        print(count)
    return task.cont

global app, ex, papp, count
count=0
app = QApplication(sys.argv)
ex = Example()
ex.hide()

papp = PandaApp()
papp.taskMgr.add(qt_update, 'debug output')
papp.run()

#p = ex.grab()
#p.save("pyqt_3d_test.png")
#app.sync()
#sys.exit(app.exec_())
#base.run()
#app.quit()