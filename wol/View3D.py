#!/usr/bin/env python

from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QColor, QSurfaceFormat, QVector2D, QVector3D, QMatrix4x4, QCursor
from PyQt5.QtWidgets import QOpenGLWidget
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
        self.scene = RootNode(self.context)
        self.scene.context.current_camera = CameraNode(self.scene)

        self.updateTimer = QTimer(self)
        self.updateTimer.setInterval(20)
        self.updateTimer.timeout.connect(self.scene_update)
        self.updateTimer.start()

        self.key_map = {
            'w': 'forward',
            's': 'back',
            'a': 'left',
            'd': 'right',
            'e': 'active_action'}
        self.setMouseTracking(True)
        self.skipNextMouseMove = True
        self.keepMouseCentered = True

    def initializeGL(self):
        GL.glEnable(GL.GL_DEPTH_TEST)
        self.program = ShadersLibrary.create_program('simple_texture')
        self.program.bind()
        self.program.setUniformValue('texture', 0)
        self.scene.initialize_gl_recurs()

    def paintGL(self):
        GL.glClearColor(self.clearColor.redF(),
                        self.clearColor.greenF(),
                        self.clearColor.blueF(),
                        self.clearColor.alphaF())
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self.scene.paint_recurs(self.program)

    def resizeGL(self, width, height):
        side = min(width, height)
        GL.glViewport((width - side) // 2, (height - side) // 2, side, side)
        print("resize")
        self.context.current_camera.ratio = width / height

    def scene_update(self):
        # Put the collision detection in a more appropriate place (RootNode? Context?)
        ray = Collisions.Ray(self.context.current_camera.position,
                             self.context.current_camera.look_at)
        colliders = list()
        for child in self.scene.children:
            if child.collider is None:
                continue
            transform = QMatrix4x4()
            transform.rotate(child.orientation)
            transform.translate(child.position)
            v, inters = Collisions.collision_ray(ray, child.collider, transform)
            if v:
                colliders.append((inters, child, (inters-self.context.current_camera.position).length()))
        colliders.sort(key=lambda s: s[2])
        if len(colliders) > 0:
            self.context.debug_point = colliders[0][0]
        else:
            self.context.debug_point = QVector3D(0, 0, -10)
        self.scene.update_recurs(0.01)
        self.repaint()

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
        if evt.isAutoRepeat():
            return
        action = self.key_map.get(evt.text(), None)
        if action:
            self.context.abstract_input[action] = True

    def keyReleaseEvent(self, evt):
        if evt.isAutoRepeat():
            return
        action = self.key_map.get(evt.text(), None)
        if action:
            self.context.abstract_input[action] = False

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

    def closeEvent(self, evt):
        self.updateTimer.stop()

    def enterEvent(self, evt):
        mid = QPoint(self.width() / 2, self.height() / 2)
        c = QCursor()
        self.skipNextMouseMove = True
        c.setPos(mid)
        self.setCursor(Qt.BlankCursor)

    def leaveEvent(self, evt):
        self.setCursor(Qt.ArrowCursor)
