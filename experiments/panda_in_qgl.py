import platform

#if platform.architecture()[0] != "32bit":
##    raise Exception("Only 32bit architecture is supported")

from pandac.PandaModules import loadPrcFileData

# loadPrcFileData("", "window-type offscreen") # Set Panda to draw its main window in an offscreen buffer
loadPrcFileData("", "load-display pandagl")
#loadPrcFileData("", "model-path c:/Panda3D-1.8.1/models")
loadPrcFileData("", "win-size 800 600")
loadPrcFileData("", "show-frame-rate-meter #t")
# PyQt imports
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QTextEdit, QMainWindow
from PyQt5.QtOpenGL import QGLWidget
from PyQt5.QtGui import QPixmap, QImage, QMouseEvent, QKeyEvent
from PyQt5.QtCore import QEvent, Qt, QPoint, QObject, QTimer

import os, sys, platform

# Panda imports
from direct.showbase.DirectObject import DirectObject
from pandac.PandaModules import WindowProperties, FrameBufferProperties, CallbackGraphicsWindow, GraphicsOutput, \
    Texture, StringStream, PNMImage
from direct.interval.LerpInterval import LerpHprInterval
from pandac.PandaModules import Point3
# Set up Panda environment
import direct.directbase.DirectStart
from struct import *

P3D_WIN_WIDTH = 800
P3D_WIN_HEIGHT = 600

class MainWindow(QMainWindow):
    mainFrame = None

    def __init__(self):
        super(MainWindow, self).__init__(None)
        self.mainFrame = EmbeddedPandaWindow()
        self.setCentralWidget(self.mainFrame)
        self.resize(800, 600)
        self.setWindowTitle('PyQT example')
        self.setFocusPolicy(Qt.StrongFocus)


class EmbeddedPandaWindow(QGLWidget):
    """ This class implements a Panda3D window that is directly
    embedded within the frame.  It is fully supported on Windows,
    partially supported on Linux, and not at all on OSX. """

    def __init__(self, *args, **kw):
        gsg = None
        if 'gsg' in kw:
            gsg = kw['gsg']
            del kw['gsg']

        fbprops = kw.get('fbprops', None)
        if fbprops == None:
            fbprops = FrameBufferProperties.getDefault()

        QGLWidget.__init__(self, *args, **kw)

        base.disableMouse()
        base.camera.setPos(0, -28, 6)
        self.testModel = loader.loadModel('panda.egg.pz')
        self.testModel.reparentTo(render)
        self.rotateInterval = LerpHprInterval(self.testModel, 3, Point3(360, 0, 0))
        self.rotateInterval.loop()

        self.callbackWindowDict = {
            'Events': self.__eventsCallback,
            'Properties': self.__propertiesCallback,
            'Render': self.__renderCallback,
        }

        # Make sure we have an OpenGL GraphicsPipe.
        if not base.pipe:
            base.makeDefaultPipe()
        self.pipe = base.pipe
        if self.pipe.getInterfaceName() != 'OpenGL':
            base.makeAllPipes()
            for self.pipe in base.pipeList:
                if self.pipe.getInterfaceName() == 'OpenGL':
                    break

        if self.pipe.getInterfaceName() != 'OpenGL':
            raise StandardError("Couldn't get an OpenGL pipe.")

        self.win = base.openWindow(callbackWindowDict=self.callbackWindowDict, pipe=self.pipe, gsg=gsg, type='none',
                                   unexposedDraw=True)

        # Setup a timer in Qt that runs taskMgr.step() to simulate Panda's own main loop
        pandaTimer = QTimer(self)
        pandaTimer.timeout.connect(taskMgr.step)
        pandaTimer.start(0)

        #base.bindToWindow(int(self.winId()))
        wp = WindowProperties().getDefault()
        wp.setOrigin(0, 0)
        wp.setSize(P3D_WIN_WIDTH, P3D_WIN_HEIGHT)
        wp.setParentWindow(int(self.winId()))
        base.openDefaultWindow(props=wp, startDirect=False)
        self.wp = wp

    def closeEvent(self, event):
        self.cleanup()
        event.Skip()

    def cleanup(self):
        """ Parent windows should call cleanup() to clean up the
        wxPandaWindow explicitly (since we can't catch EVT_CLOSE
        directly). """
        if self.win:
            base.closeWindow(self.win)
            self.win = None

    def onSize(self, event):
        wp = WindowProperties()
        wp.setOrigin(0, 0)
        wp.setSize(*self.GetClientSize())
        self.win.requestProperties(wp)
        event.Skip()

    def __eventsCallback(self, data):
        data.upcall()

    def __propertiesCallback(self, data):
        data.upcall()

    def __renderCallback(self, data):
        cbType = data.getCallbackType()
        if cbType == CallbackGraphicsWindow.RCTBeginFrame:
            if not self.isVisible():
                data.setRenderFlag(False)
                return
            self.updateGL()

            # Don't upcall() in this case.
            return

        elif cbType == CallbackGraphicsWindow.RCTEndFlip:
            # Now that we've swapped, ask for a refresh, so we'll
            # get another paint message if the window is still
            # visible onscreen.
            self.updateGL()

        data.upcall()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    app.exec_()

    # Need to destroy QApplication(), otherwise Shutdown() fails.
    # Unset main window also just to be safe.
    del mainWindow
    del app

    sys.exit(0)