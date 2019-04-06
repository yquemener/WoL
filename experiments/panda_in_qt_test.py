# -*- coding: utf-8-*-

from pandac.PandaModules import loadPrcFileData

loadPrcFileData("", "window-type none")

from direct.showbase.DirectObject import DirectObject
from pandac.PandaModules import WindowProperties

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtOpenGL import QGLWidget

import sys

P3D_WIN_WIDTH = 400
P3D_WIN_HEIGHT = 240

from panda3d.core import Texture, CardMaker, TextureStage, MouseWatcher
from direct.showbase.ShowBase import ShowBase

global renderCounter
renderCounter=0

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
        self.ex = None
        base.accept('setEventsCallback', self.printHello)

    def printHello(self):
        print('Hello')

    def bindToWindow(self, windowHandle, widget):
        wp = WindowProperties().getDefault()
        wp.setOrigin(0, 0)
        wp.setSize(P3D_WIN_WIDTH, P3D_WIN_HEIGHT)
        wp.setParentWindow(windowHandle)
        self.parent_widget = widget
        base.openDefaultWindow(props=wp, startDirect=True,
                               callbackWindowDict={
                                   #'Events': self.__eventsCallback,
                                   'Properties': self.__propertiesCallback,
                                   #'Render': self.__renderCallback,
                               },
                               unexposedDraw=True)
        self.wp = wp


    def __eventsCallback(self, data):
        #print(data)
        data.upcall()

    def __propertiesCallback(self, data):
        data.upcall()

    def __renderCallback(self, data):
        global renderCounter
        renderCounter+=1
        print(renderCounter)
        self.parent_widget.makeCurrent()
        data.upcall()
        self.parent_widget.updateGL()


    def step(self):
        taskMgr.step()
        i = QImage(self.ex.grab())
        buf = i.bits().asstring(250 * 250 * 4)
        self.tex.setRamImage(buf)
        self.card.setTexture(self.tex)

class QTTest(QDialog):
    def __init__(self, pandaCallback, parent=None):
        super(QDialog, self).__init__(parent)
        self.setWindowTitle("Test")
        self.setGeometry(0, 0, 400, 300)

        self.pandaContainer = QGLWidget(self)
        self.pandaContainer.setGeometry(0, 0, P3D_WIN_WIDTH, P3D_WIN_HEIGHT)

        self.lineedit = QLineEdit("Write something...does it work?")

        layout = QVBoxLayout()
        layout.addWidget(self.pandaContainer)
        layout.addWidget(self.lineedit)

        self.setLayout(layout)

        self.ex = Example()
        self.ex.hide()

        # this basically creates an idle task
        timer = QTimer(self)
        timer.timeout.connect(pandaCallback)
        #self.connect(timer, SIGNAL("timeout()"), pandaCallback)
        timer.start(0)

global app, form

class myEventFilter(QObject):
    def eventFilter(self, obj, evt):
        global app, form
        #print("filter: ", QEvent.Type(evt.type()), evt, obj)
        if evt.type() not in (QEvent.Paint, QEvent.Timer, 43, 77,QEvent.MouseMove):
            #print("filter: ", QEvent.Type(evt.type()), evt, obj)
            pass
        #if evt.type() in (QEvent.InputMethodEvent,):
        #    if form.ex.edit is not None:
        #        form.ex.edit.inputMethodEvent(evt)
        #        return True

        return QObject.eventFilter(self, obj, evt)


class Example(QWidget):
    def __init__(self):
        super(Example, self).__init__()
        self.inp
        self.initUI()

    def initUI(self):
        qbtn = QPushButton('Quit', self)
        print(type(qbtn.clicked))
        qbtn.resize(qbtn.sizeHint())
        qbtn.move(50, 50)
        qedit = QTextEdit('TestEdit', self)
        qedit.resize(qedit.sizeHint())
        qedit.move(50, 80)
        self.qbtn2 = QPushButton('+', self)
        self.qbtn2.resize(qbtn.sizeHint())
        self.button = qbtn
        self.edit = qedit

        self.setGeometry(300, 300, 250, 250)
        self.setWindowTitle('Quit button')
        print("grab")
        self.edit.grabKeyboard()
        self.show()

    def debugEvent(self, e):
        print(e)
        self.edit.origFunc(e)

    def mousePressEvent(self, evt):
        print(evt.pos())
        self.qbtn2.move(evt.pos())


if __name__ == '__main__':
    world = PandaApp()

    app = QApplication(sys.argv)
    form = QTTest(world.step)
    #form.installEventFilter(myEventFilter(app))
    #form.lineedit.installEventFilter(myEventFilter(app))
    form.pandaContainer.installEventFilter(myEventFilter(app))
    world.ex = form.ex
    world.bindToWindow(int(form.pandaContainer.winId()), form.pandaContainer)

    form.show()
    app.exec_()