#!/usr/bin/env python

from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QColor, QSurfaceFormat, QVector2D, QVector3D, QMatrix4x4, QCursor, QMouseEvent, QKeyEvent
from PyQt5.QtWidgets import QOpenGLWidget, QApplication
from OpenGL import GL

from wol.Context import Context
from wol.ShadersLibrary import ShadersLibrary
from wol.SceneNode import RootNode, CameraNode
import wol.Collisions as Collisions


class View3D(QOpenGLWidget):
    def __init__(self, parent=None):
        super(View3D, self).__init__(parent)

        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        QSurfaceFormat.setDefaultFormat(fmt)

        self.clearColor = QColor(Qt.black)
        self.program = None
        self.context = Context()
        self.context.scene = RootNode(self.context)

        self.updateTimer = QTimer(self)
        self.updateTimer.setInterval(20)
        self.updateTimer.timeout.connect(self.scene_update)
        self.updateTimer.start()

        self.key_map = {
            Qt.Key_W: 'forward',
            Qt.Key_S: 'back',
            Qt.Key_A: 'left',
            Qt.Key_D: 'right',
            Qt.Key_Space: 'up',
            Qt.Key_Shift: "down"}
        self.setMouseTracking(True)
        self.skipNextMouseMove = True
        self.keepMouseCentered = True
        self.setAttribute(Qt.WA_InputMethodEnabled, True)
        self.setGeometry(10, 10, 1200, 800)

    def initializeGL(self):
        GL.glEnable(GL.GL_DEPTH_TEST)
        self.program = ShadersLibrary.create_program('simple_texture')
        self.program.bind()
        self.program.setUniformValue('texture', 0)
        self.context.scene.initialize_gl_recurs()

    def paintGL(self):
        GL.glClearColor(self.clearColor.redF(),
                        self.clearColor.greenF(),
                        self.clearColor.blueF(),
                        self.clearColor.alphaF())
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        for layer in (0, 1, 2, 3):
            self.context.scene.paint_recurs(self.program, layer)

    def resizeGL(self, width, height):
        side = min(width, height)
        GL.glViewport((width - side) // 2, (height - side) // 2, side, side)
        self.context.current_camera.ratio = width / height

    def scene_update(self):
        self.context.scene.update_recurs(0.01)
        # Put the collision detection in a more appropriate place (RootNode? Context?)
        ray = Collisions.Ray(self.context.current_camera.position,
                             self.context.current_camera.look_at-self.context.current_camera.position)
        colliders = self.context.scene.collide_recurs(ray)
        colliders_sort = list()
        for obj, point in colliders:
            colliders_sort.append((point,
                                   obj,
                                   (point-self.context.current_camera.position).length()))
        colliders_sort.sort(key=lambda s: s[2])
        if len(colliders_sort) > 0:
            self.context.debug_point = colliders_sort[0][0]
            self.context.hover_target = colliders_sort[0][1]
        else:
            self.context.debug_point = QVector3D(0, 0, -10)
            self.context.hover_target = None
        # Ugly
        self.context.scene.sphere.position = self.context.debug_point
        self.repaint()

    def keyReleaseEvent(self, evt):
        if evt.isAutoRepeat():
            return
        action = self.key_map.get(evt.key(), None)
        if action:
            self.context.abstract_input[action] = False

    def keyPressEvent(self, evt):
        if evt.key() == Qt.Key_Escape:
            self.close()
        if evt.key() == Qt.Key_Tab:
            self.releaseMouse()
            self.keepMouseCentered = not self.keepMouseCentered
            if self.keepMouseCentered:
                self.setCursor(Qt.BlankCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        if self.context.focused is not None:
            self.context.focused.keyPressEvent(evt)
            return
        if evt.key() == Qt.Key_E:
            if hasattr(self.context.hover_target, "on_edit"):
                self.context.hover_target.on_edit(self.context.debug_point)

        if evt.key() == Qt.Key_R:
            if hasattr(self.context.hover_target, "on_save"):
                self.context.hover_target.on_save(self.context.debug_point)

        if evt.key() == Qt.Key_O:
            self.saveScene()

        if evt.key() == Qt.Key_Q:
            if self.context.hover_target:
                gr = self.context.grabbed
                if gr is None:
                    self.context.grabbed = self.context.hover_target
                    self.context.grabbed_former_parent = self.context.grabbed.parent
                    self.context.grabbed.reparent(self.context.current_camera)
                else:
                    self.context.grabbed.reparent(self.context.grabbed_former_parent)
                    self.context.grabbed = None

        if evt.isAutoRepeat():
            return
        action = self.key_map.get(evt.key(), None)
        if action:
            self.context.abstract_input[action] = True

    def saveScene(self):
        scenefile = open("scene.ini", "w")
        s = ""
        for c in self.context.scene.children:
            s += c.serialize_recurs()
        scenefile.write(s)

    def inputMethodEvent(self, evt):
        if self.context.focused is not None:
            return self.context.focused.inputMethodEvent(evt)

    def mouseMoveEvent(self, evt):
        if self.skipNextMouseMove or not self.keepMouseCentered:
            self.skipNextMouseMove = False
            return
        mid = QVector2D(self.width()/2, self.height()/2)
        delta = (1.0/self.width()) * (mid - QVector2D(evt.pos().x(), evt.pos().y()))
        c = QCursor()
        self.skipNextMouseMove = True
        c.setPos(QPoint(int(mid.x()), int(mid.y()))+self.geometry().topLeft())
        self.context.mouse_position += delta
        y = self.context.mouse_position.y()
        self.context.mouse_position.setY(max(-0.95, min(0.95, y)))

    def mousePressEvent(self, evt):
        if evt.button() == Qt.LeftButton:
            if self.context.hover_target is not None:
                self.context.hover_target.on_click(self.context.debug_point, evt)
        if evt.button() == Qt.RightButton:
            if self.context.focused is not None:
                self.context.focused.on_unfocus()
                self.context.focused = None

    def closeEvent(self, evt):
        self.updateTimer.stop()

    def enterEvent(self, evt):
        mid = QPoint(self.pos().x() + self.width() / 2, self.pos().y() + self.height() / 2)
        c = QCursor()
        self.skipNextMouseMove = True
        c.setPos(mid)
        self.setCursor(Qt.BlankCursor)

    def leaveEvent(self, evt):
        self.setCursor(Qt.ArrowCursor)
