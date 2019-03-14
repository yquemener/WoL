#!/usr/bin/env python

import sys

from PyQt5.QtCore import pyqtSignal, QFileInfo, QPoint, QSize, Qt, QTimer, QEvent, QObject
from PyQt5.QtGui import (QColor, QImage, QMatrix4x4, QOpenGLShader,
        QOpenGLShaderProgram, QOpenGLTexture, QSurfaceFormat, QInputMethodEvent)
from PyQt5.QtWidgets import QApplication, QGridLayout, QOpenGLWidget, QWidget, QPushButton, QTextEdit
from OpenGL import GL

class GLWidget(QOpenGLWidget):

    clicked = pyqtSignal()

    PROGRAM_VERTEX_ATTRIBUTE, PROGRAM_TEXCOORD_ATTRIBUTE = range(2)

    vsrc = """
attribute highp vec4 vertex;
attribute mediump vec4 texCoord;
varying mediump vec4 texc;
uniform mediump mat4 matrix;
void main(void)
{
    gl_Position = matrix * vertex;
    texc = texCoord;
}
"""

    fsrc = """
uniform sampler2D texture;
varying mediump vec4 texc;
void main(void)
{
    gl_FragColor = texture2D(texture, texc.st);
}
"""

    coords = (
        (( +1, -1, -1 ), ( -1, -1, -1 ), ( -1, +1, -1 ), ( +1, +1, -1 )),
        (( +1, +1, -1 ), ( -1, +1, -1 ), ( -1, +1, +1 ), ( +1, +1, +1 )),
        (( +1, -1, +1 ), ( +1, -1, -1 ), ( +1, +1, -1 ), ( +1, +1, +1 )),
        (( -1, -1, -1 ), ( -1, -1, +1 ), ( -1, +1, +1 ), ( -1, +1, -1 )),
        (( +1, -1, +1 ), ( -1, -1, +1 ), ( -1, -1, -1 ), ( +1, -1, -1 )),
        (( -1, -1, +1 ), ( +1, -1, +1 ), ( +1, +1, +1 ), ( -1, +1, +1 ))
    )

    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)

        self.clearColor = QColor(Qt.black)
        self.xRot = 0
        self.yRot = 0
        self.zRot = 0
        self.program = None

        self.lastPos = QPoint()
        self.ex = Example()

    def minimumSizeHint(self):
        return QSize(50, 50)

    def sizeHint(self):
        return QSize(200, 200)

    def rotateBy(self, xAngle, yAngle, zAngle):
        self.xRot += xAngle
        self.yRot += yAngle
        self.zRot += zAngle
        self.update()

    def setClearColor(self, color):
        self.clearColor = color
        self.update()

    def initializeGL(self):
        #GL = self.context().versionFunctions()
        #GL.initializeOpenGLFunctions()

        self.makeObject()

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)

        vshader = QOpenGLShader(QOpenGLShader.Vertex, self)
        vshader.compileSourceCode(self.vsrc)

        fshader = QOpenGLShader(QOpenGLShader.Fragment, self)
        fshader.compileSourceCode(self.fsrc)

        self.program = QOpenGLShaderProgram()
        self.program.addShader(vshader)
        self.program.addShader(fshader)
        self.program.bindAttributeLocation('vertex',
                self.PROGRAM_VERTEX_ATTRIBUTE)
        self.program.bindAttributeLocation('texCoord',
                self.PROGRAM_TEXCOORD_ATTRIBUTE)
        print(self.program.link(), self.PROGRAM_VERTEX_ATTRIBUTE)

        self.program.bind()
        self.program.setUniformValue('texture', 0)

        self.program.enableAttributeArray(self.PROGRAM_VERTEX_ATTRIBUTE)
        self.program.enableAttributeArray(self.PROGRAM_TEXCOORD_ATTRIBUTE)
        self.program.setAttributeArray(self.PROGRAM_VERTEX_ATTRIBUTE,
                self.vertices)
        self.program.setAttributeArray(self.PROGRAM_TEXCOORD_ATTRIBUTE,
                self.texCoords)

    def paintGL(self):
        i = QImage(self.ex.grab()).mirrored()
        self.textures[0] = QOpenGLTexture(i)


        GL.glClearColor(self.clearColor.redF(), self.clearColor.greenF(),
                self.clearColor.blueF(), self.clearColor.alphaF())
        GL.glClear(
                GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        m = QMatrix4x4()
        m.ortho(-0.5, 0.5, 0.5, -0.5, 4.0, 15.0)
        m.translate(0.0, 0.0, -10.0)
        m.rotate(self.xRot / 16.0, 1.0, 0.0, 0.0)
        m.rotate(self.yRot / 16.0, 0.0, 1.0, 0.0)
        m.rotate(self.zRot / 16.0, 0.0, 0.0, 1.0)

        self.program.setUniformValue('matrix', m)

        for i, texture in enumerate(self.textures):
            texture.bind()
            GL.glDrawArrays(GL.GL_TRIANGLE_FAN, i * 4, 4)

    def resizeGL(self, width, height):
        side = min(width, height)
        GL.glViewport((width - side) // 2, (height - side) // 2, side,
                side)

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & Qt.LeftButton:
            self.rotateBy(8 * dy, 8 * dx, 0)
        elif event.buttons() & Qt.RightButton:
            self.rotateBy(8 * dy, 0, 8 * dx)

        self.lastPos = event.pos()

    def mouseReleaseEvent(self, event):
        self.clicked.emit()

    def makeObject(self):
        self.textures = []
        self.texCoords = []
        self.vertices = []

        root = QFileInfo(__file__).absolutePath()

        for i in range(6):
            self.textures.append(
                    QOpenGLTexture(
                            QImage('/home/yves/path4787.png').mirrored()))

            for j in range(4):
                self.texCoords.append(((j == 0 or j == 3), (j == 0 or j == 1)))

                x, y, z = self.coords[i][j]
                self.vertices.append((0.2 * x, 0.2 * y, 0.2 * z))



class Example(QWidget):
    def __init__(self):
        super(Example, self).__init__()

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
        #self.show()

    def debugEvent(self, e):
        print(e)
        self.edit.origFunc(e)

    def mousePressEvent(self, evt):
        print(evt.pos())
        self.qbtn2.move(evt.pos())


class Window(QWidget):
    NumRows = 2
    NumColumns = 3

    def __init__(self):
        super(Window, self).__init__()

        mainLayout = QGridLayout()

        clearColor = QColor()
        clearColor.setHsv(64, 255, 63)

        widget = GLWidget()
        widget.setClearColor(clearColor)
        #widget.rotateBy(+42 * 16, +42 * 16, -21 * 16)
        mainLayout.addWidget(widget, 0, 0)

        widget.clicked.connect(self.setCurrentGlWidget)

        self.setLayout(mainLayout)

        self.currentGlWidget = widget

        timer = QTimer(self)
        timer.timeout.connect(self.currentGlWidget.update)
        timer.start(20)

        self.setWindowTitle("Textures")
        self.setAttribute(Qt.WA_InputMethodEnabled, True)

    def setCurrentGlWidget(self):
        self.currentGlWidget = self.sender()

    def rotateOneStep(self):
        if self.currentGlWidget:
            self.currentGlWidget.rotateBy(+2 * 16, +2 * 16, -1 * 16)

    def inputMethodQueryEvent(self, a):
         print("query", a)

    def inputMethodEvent(self, a):
        i = QInputMethodEvent()
        print("a", a, dir(a.InputMethodQuery))
        return self.currentGlWidget.ex.edit.inputMethodEvent(a)


class myEventFilter(QObject):
    def eventFilter(self, obj, evt):
        if evt.type() not in (QEvent.Paint, QEvent.Timer, 77, QEvent.MouseMove):
            print("filter: ", QEvent.Type(evt.type()), evt, obj)
            #print("filter: ", QEvent.Type(evt.type()), evt, obj)
            pass
        #if evt.type() in (QEvent.InputMethodEvent,):
        #    if form.ex.edit is not None:
        #        form.ex.edit.inputMethodEvent(evt)
        #        return True

        return QObject.eventFilter(self, obj, evt)

if __name__ == '__main__':

    app = QApplication(sys.argv)

    format = QSurfaceFormat()
    format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(format)

    window = Window()
    window.show()
    window.currentGlWidget.ex.edit.installEventFilter(myEventFilter(app))
    sys.exit(app.exec_())
