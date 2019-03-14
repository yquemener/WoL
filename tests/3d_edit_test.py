#!/usr/bin/python3
# -*- coding: utf-8 -*-


import sys
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QTextEdit
from PyQt5.QtGui import QPixmap, QImage, QMouseEvent, QKeyEvent
from PyQt5.QtCore import QEvent, Qt, QPoint, QObject

from panda3d.core import Texture, CardMaker, TextureStage, MouseWatcher
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
        #self.accept(PGEntry().getTypeEvent(), self._handleTyping)

    def _handleTyping(self, evt):
        print(evt)
        print(dir(evt))


class Example(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def debugClick(self, *args, **kwargs):
        print(args, kwargs)

    def initUI(self):
        qbtn = QPushButton('Quit', self)
        print(type(qbtn.clicked))
        qbtn.clicked.connect(self.debugClick)
        qbtn.resize(qbtn.sizeHint())
        qbtn.move(50, 50)
        qedit = QTextEdit('TestEdit', self)
        qedit.resize(qedit.sizeHint())
        qedit.move(50, 80)
        qedit.textChanged.connect(self.debugClick)
        self.qbtn2 = QPushButton('+', self)
        self.qbtn2.resize(qbtn.sizeHint())
        self.button = qbtn
        self.edit = qedit

        self.setGeometry(300, 300, 250, 250)
        self.setWindowTitle('Quit button')
        #self.edit.keyPressEvent = self.debugClick
        self.edit.grabKeyboard()
        self.show()

    def mousePressEvent(self, evt):
        print(evt.pos())
        self.qbtn2.move(evt.pos())


class myEventFilter(QObject):
    def eventFilter(self, obj, evt):
        if evt.type() not in (QEvent.Paint, QEvent.Timer, 43, 77,QEvent.MouseMove):
            print("filter: ", QEvent.Type(evt.type()), evt, obj)
        return super().eventFilter(obj, evt)


def qt_update(task):
    global app, ex, papp, count
    app.processEvents()
    i = QImage(ex.grab())
    buf = i.bits().asstring(250*250*4)
    papp.tex.setRamImage(buf)
    papp.card.setTexture(papp.tex)
    count += 1
    clic_pos = QPoint(96, 60)

    child = ex.childAt(clic_pos)
    if child is None:
        child = ex
    else:
        clic_pos -= child.pos()
    if count % 200 == 0:
        e = QMouseEvent(QMouseEvent.MouseButtonPress,
                        clic_pos,
                        Qt.LeftButton,
                        Qt.LeftButton,
                        Qt.NoModifier)
        #QApplication.instance().postEvent(child, e)
        k = QKeyEvent(QKeyEvent.KeyPress,
                      Qt.Key_A,
                      Qt.NoModifier,
                      text='a')
        #QApplication.instance().sendEvent(ex.edit, k)
    if count % 200 == 155:
        e = QMouseEvent(QMouseEvent.MouseButtonRelease,
                        clic_pos,
                        Qt.LeftButton,
                        Qt.LeftButton,
                        Qt.NoModifier)
        #QApplication.instance().postEvent(child, e)
        k = QKeyEvent(QKeyEvent.KeyRelease,
                      Qt.Key_A,
                      Qt.NoModifier,
                      text='a')
        #QApplication.instance().sendEvent(ex.edit, k)
    #print(app.widgetAt(ex.pos().x()+6, ex.pos().y()+60))
    return task.cont


global app, ex, papp, count
count = 0
app = QApplication(sys.argv)
app.installEventFilter(myEventFilter(app))
ex = Example()

#ex.hide()

papp = PandaApp()
papp.taskMgr.add(qt_update, 'debug output')
papp.run()

#p = ex.grab()
#p.save("pyqt_3d_test.png")
#app.sync()
#sys.exit(app.exec_())
#base.run()
#app.quit()