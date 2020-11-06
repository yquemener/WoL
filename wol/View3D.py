#!/usr/bin/env python

from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QColor, QSurfaceFormat, QVector2D, QVector3D, QCursor, QVector4D, QMatrix3x3, QQuaternion, \
    QMatrix4x4
from PyQt5.QtWidgets import QOpenGLWidget
from OpenGL import GL
from math import sin

from wol.ConsoleNode import ConsoleNode
from wol.GuiElements import TextLabelNode
from wol.PlayerContext import PlayerContext
from wol.ShadersLibrary import ShadersLibrary
from wol.SceneNode import RootNode, SceneNode
import wol.Collisions as Collisions


class View3D(QOpenGLWidget):
    def __init__(self, parent=None):
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setSamples(8)
        QSurfaceFormat.setDefaultFormat(fmt)
        super(View3D, self).__init__(parent)


        self.clearColor = QColor(Qt.black)
        self.program = None
        self.context = PlayerContext()
        self.context.scene = RootNode(self.context)
        self.context.hud = list()
        self.real_mouse_position = (0, 0)

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

        # HUD definition
        hud1 = TextLabelNode(parent=None, text="Salut!")
        hud1.position.z = 0
        for v in hud1.vertices:
            v[1] *= -600
            v[1] += 100
            v[0] *= -400
            v[0] += 100
        hud1.refresh_vertices()
        self.context.hud.append(hud1)


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

        # Add HUD/GUI layer here
        GL.glDisable(GL.GL_DEPTH_TEST)
        w = self.geometry().width()
        h = self.geometry().height()
        mat = QMatrix4x4([2. / w,    0.0,      -1,      0,
                          0.0,       -2. / h,  1,       0,
                          0.0,       0.0,      0.0,     0,
                          0,         0,         0,      1])
        # Draw crosshair
        crosshair = [QVector3D(w / 2, h / 2 - 20,   0),
                     QVector3D(w / 2, h / 2 - 5,    0),
                     QVector3D(w / 2, h / 2 + 20,   0),
                     QVector3D(w / 2, h / 2 + 5,    0),

                     QVector3D(w / 2 - 20, h / 2,   0),
                     QVector3D(w / 2 - 5, h / 2,    0),
                     QVector3D(w / 2 + 20, h / 2,   0),
                     QVector3D(w / 2 + 5, h / 2,    0)]
        crosshair_color = QVector4D(1.0, 1.0, 1.0, 0.5)
        program = ShadersLibrary.create_program('hud_2d')
        program.bind()
        program.setAttributeArray(0, crosshair)
        transmat = QMatrix4x4()
        transmat.setToIdentity()
        program.setUniformValue('z_order', 1.)
        # program.setUniformValue('transform_matrix', transmat)
        program.setUniformValue('matrix', mat)
        program.setUniformValue('material_color', crosshair_color)

        GL.glDrawArrays(GL.GL_LINES, 0, 8)
        program.setUniformValue('material_color', QVector4D(0.0, 0.0, 0.0, 0.5))
        transmat = QMatrix4x4([1.0,    0.0,      1,     0.0,
                               0.0,    1.0,      1,     0.0,
                               0.0,    0.0,      1.0,   0.0,
                               0.0,    0.0,      0.0,   1.0])
        program.setUniformValue('matrix', mat * transmat)
        GL.glDrawArrays(GL.GL_LINES, 0, 8)

        program = ShadersLibrary.create_program('hud_2d_tex')
        program.bind()
        program.setUniformValue('z_order', 1.)
        program.setUniformValue('matrix', mat)
        program.setUniformValue('material_color', QVector4D(0.0, 0.0, 0.0, 0.5))

        for h in self.context.hud:
            h.update(0)
            h.prog_matrix = mat
            h.initialize_gl()
            h.paint(program)


        GL.glEnable(GL.GL_DEPTH_TEST)

    def resizeGL(self, width, height):
        side = min(width, height)
        GL.glViewport((width - side) // 2, (height - side) // 2, side, side)
        self.context.current_camera.ratio = width / height

    def scene_update(self):
        self.context.scene.update_recurs(0.01)


        # Put the collision detection in a more appropriate place (RootNode? Context?)
        if self.keepMouseCentered:
            ray = Collisions.Ray(self.context.current_camera.position,
                                 self.context.current_camera.look_at-self.context.current_camera.position)
        else:
            #self.context.debug_sphere.parent = self.context.current_camera
            fov = 45.0*3.14159/180.0
            offx = sin(fov/2.)
            offy = offx/self.context.current_camera.ratio
            target = QVector3D(offx-self.real_mouse_position[0]/self.width()*2.*offx,
                               offy-self.real_mouse_position[1]/self.height()*2.*offy,
                               1.0)
            target = self.context.current_camera.orientation.rotatedVector(target)
            ray = Collisions.Ray(self.context.current_camera.position,
                                 target)

        colliders = self.context.scene.collide_recurs(ray)
        colliders_sort = list()
        for obj, point in colliders:
            colliders_sort.append((point,
                                   obj,
                                   (point-self.context.current_camera.position).length()))
        colliders_sort.sort(key=lambda s: s[2])
        if len(colliders_sort) > 0:
            self.context.debug_point = colliders_sort[0][0]
            # self.context.debug_sphere.position = self.context.debug_point
            self.context.hover_target = colliders_sort[0][1]
        else:
            self.context.debug_point = QVector3D(0, 0, -10)
            self.context.hover_target = None
        self.repaint()

    def keyReleaseEvent(self, evt):
        if evt.isAutoRepeat():
            return
        action = self.key_map.get(evt.key(), None)
        if action:
            self.context.abstract_input[action] = False

    def keyPressEvent(self, evt):
        #if evt.key() == Qt.Key_Escape:
        #    self.close()

        if evt.key() == Qt.Key_Escape:
            if self.context.focused is not None:
                self.context.focused.focused = False
                self.context.focused = None
                stc = self.context.current_camera.get_behavior("SnapToCamera")
                if stc.grabbed_something:
                    stc.restore()

        if self.context.focused is not None:
            self.context.focused.keyPressEvent(evt)
            return

        if evt.key() == Qt.Key_Tab:
            self.releaseMouse()
            self.keepMouseCentered = not self.keepMouseCentered
            if self.keepMouseCentered:
                self.setCursor(Qt.BlankCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

        if evt.key() == Qt.Key_QuoteLeft:
            stc = self.context.current_camera.get_behavior("SnapToCamera")

            if not hasattr(self.context, "current_console"):
                console = None
                for o in self.context.scene.children:
                    if isinstance(o, ConsoleNode):
                        console = o
                if console is None:
                    console = ConsoleNode(parent=self.context.scene)
                self.context.current_console = console

            if stc.grabbed_something:
                if stc.target is self.context.current_console:
                    stc.restore()
                    self.context.focused = None
                    self.context.current_console.focused = False

                    return
                stc.restore()
            stc.grab(self.context.current_console)
            if self.context.focused:
                self.context.focused.focused = False
            self.context.focused = self.context.current_console
            self.context.current_console.focused = True

        if evt.key() == Qt.Key_E:
            if hasattr(self.context.hover_target, "on_edit"):
                self.context.hover_target.on_edit(self.context.debug_point)

        if evt.key() == Qt.Key_T:
            stc = self.context.current_camera.get_behavior("SnapToCamera")
            if not stc.grabbed_something:
                if self.context.hover_target is not None:
                    stc.grab(self.context.hover_target)
            else:
                stc.restore()

        if evt.key() == Qt.Key_R:
            s = "import wol\nimport PyQt5\n\n"

            num = 0
            for o in self.context.scene.children:
                ss, num = o.serialize(num)
                s += ss
            print(s)
            fout = open("init_scene.py", "w")
            fout.write(s)
            fout.close()
            if hasattr(self.context.hover_target, "on_save"):
                self.context.hover_target.on_save(self.context.debug_point)

        if evt.key() == Qt.Key_L:
            fout = open("init_scene.py", "r")
            s = fout.read()
            fout.close()
            d = globals()
            d["context"]=self.context
            exec(s, d, d)

        if evt.key() == Qt.Key_O:
            self.saveScene()

        if evt.key() == Qt.Key_1:
            self.snap_to_90()

        if evt.key() == Qt.Key_Q:
            if self.context.hover_target:
                gr = self.context.grabbed
                if gr is None:
                    anchor = self.context.hover_target
                    while anchor.properties.get("delegateGrabToParent", False) \
                            and anchor.parent is not None \
                            and anchor.parent is not self.context.scene:
                        anchor = anchor.parent
                    self.context.grabbed = anchor
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

    def snap_to_90(self):
        if self.context.grabbed is None:
            return
        g = self.context.grabbed
        euler = g.world_orientation().toEulerAngles()
        euler.setX(round(euler.x() / 90) * 90)
        euler.setY(round(euler.y() / 90) * 90)
        euler.setZ(round(euler.z() / 90) * 90)
        g.set_world_orientation(QQuaternion.fromEulerAngles(euler))
        # g.set_world_orientation(g.world_orientation())

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
        self.real_mouse_position = (evt.x(), evt.y())
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
            target = self.context.hover_target
            if target is not None:
                target.on_click(self.context.debug_point, evt)
                for b in target.behaviors:
                    b.on_click(self.context.debug_point, evt)
                if hasattr(target, 'focusable') and target.focusable:
                    if self.context.focused:
                        self.context.focused.focused = False
                    self.context.focused = target
                    target.focused = True
                    stc = self.context.current_camera.get_behavior("SnapToCamera")
                    if stc.grabbed_something:
                        stc.restore()
                    stc.grab(target)

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
