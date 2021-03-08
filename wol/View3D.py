#!/usr/bin/env python
from collections import defaultdict

from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QColor, QSurfaceFormat, QVector2D, QVector3D, QCursor, QVector4D, QMatrix3x3, QQuaternion, \
    QMatrix4x4
from PyQt5.QtWidgets import QOpenGLWidget, QApplication
from OpenGL import GL
from math import sin

from wol.Constants import Events, UserActions
from wol.GuiElements import TextLabelNode
from wol.PlayerContext import PlayerContext, MappingTypes
from wol.ShadersLibrary import ShadersLibrary
from wol.SceneNode import RootNode, SceneNode
import wol.Collisions as Collisions
from wol.utils import DotDict


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
        self.scene = RootNode(self.context)
        self.context.scene = self.scene
        self.hud = DotDict()
        self.hud_root = RootNode(self.context)
        # self.hud_root.orientation = QQuaternion.fromAxisAndAngle(0, 1, 0, 180)
        self.real_mouse_position = (0, 0)
        self.events_handlers = defaultdict(list)
        self.events_handlers[UserActions.Save].append(self.save_scene)
        self.events_handlers[UserActions.Load].append(self.load_scene)
        self.events_handlers[UserActions.Change_Cursor_Mode].append(self.toggle_mouse_captive)

        self.updateTimer = QTimer(self)
        self.updateTimer.setInterval(20)
        self.updateTimer.timeout.connect(self.scene_update)
        self.updateTimer.start()

        self.key_pressed = set()
        self.setMouseTracking(True)
        self.skipNextMouseMove = True
        self.keepMouseCentered = True
        self.setAttribute(Qt.WA_InputMethodEnabled, True)

        screenRect = QApplication.desktop().screenGeometry(0)

        self.setGeometry(10, 10, 1200, 800)
        self.move(QPoint(screenRect.x()+150, screenRect.y()+50))

        # HUD definition
        self.hud.hud1 = TextLabelNode(parent=self.hud_root, text="")
        self.hud.hud1.layer = -1
        self.hud.hud1.position.setX(0.5)
        self.hud.hud1.position.setY(0.5)

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

        # Draw crosshair
        crosshair = [QVector3D(0, -0.05,   0),
                     QVector3D(0, -0.01,   0),
                     QVector3D(0, 0.05,   0),
                     QVector3D(0, 0.01,   0),

                     QVector3D(-0.04, 0,   0),
                     QVector3D(-0.01, 0,   0),
                     QVector3D(0.04,  0,   0),
                     QVector3D(0.01,  0,   0)]
        crosshair_color = QVector4D(1.0, 1.0, 1.0, 0.5)
        program = ShadersLibrary.create_program('hud_2d')
        program.bind()
        program.setAttributeArray(0, crosshair)
        transmat = QMatrix4x4()
        transmat.setToIdentity()
        program.setUniformValue('z_order', 1.)

        program.setUniformValue('matrix', QMatrix4x4())
        program.setUniformValue('material_color', crosshair_color)
        GL.glDrawArrays(GL.GL_LINES, 0, 8)

        program = ShadersLibrary.create_program('hud_2d_tex')
        program.bind()
        program.setUniformValue('z_order', 1.)
        # program.setUniformValue('matrix', transmat)

        w = self.geometry().width()
        h = self.geometry().height()
        if w>h:
            w = h/w
            h = 1
        else:
            h = w/h
            w = 1
        self.hud_root.scale.setX(w)
        self.hud_root.scale.setY(h)
        # pointed = str(type(self.context.hover_target))
        # if pointed != self.hud.hud1.text:
        #     name = " "
        #     if self.context.hover_target is None:
        #         pointed = " "
        #     else:
        #         if name is not None:
        #             name = str(self.context.hover_target.name)

        if self.context.hover_target is not None:
            if self.context.hover_target.tooltip is not None:
                self.hud.hud1.set_text(str(self.context.hover_target.tooltip))
            else:
                self.hud.hud1.set_text("")
        else:
            self.hud.hud1.set_text("")
        self.hud_root.paint_recurs(program, -1)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def resizeGL(self, width, height):
        side = min(width, height)
        GL.glViewport((width - side) // 2, (height - side) // 2, side, side)
        self.context.current_camera.ratio = width / height

    def scene_update(self):
        for k in self.key_pressed:
            actions = self.context.mappings.get((MappingTypes.KeyPressed, k), [])
            for action in actions:
                self.context.current_camera.on_event(action)
                if self.context.hover_target is not None:
                    self.context.hover_target.on_event(action)
        self.context.scene.lock.acquire()
        self.context.scene.update_recurs(0.01)
        self.context.scene.lock.release()
        self.hud_root.update_recurs(0.01)

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
            if obj is not self.context.grabbed:
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
        try:
            self.key_pressed.remove(evt.key())
        except KeyError:
            pass

    def on_event(self, action):
        for h in self.events_handlers[action]:
            h()

    def keyPressEvent(self, evt):
        actions = self.context.mappings.get((MappingTypes.KeyDown, evt.key()), [])

        if self.context.focused is not None:
            try:
                self.context.focused.keyPressEvent(evt)
                if UserActions.Release not in actions:
                    return
            except AttributeError:
                pass

        for a in actions:
            self.context.current_camera.on_event(a)
            self.on_event(a)
            if self.context.focused is not None:
                self.context.focused.on_event(a)
            elif self.context.grabbed is not None:
                self.context.grabbed.on_event(a)
            elif self.context.hover_target is not None:
                self.context.hover_target.on_event(a)


        if evt.isAutoRepeat():
            return
        self.key_pressed.add(evt.key())

    def toggle_mouse_captive(self):
        self.releaseMouse()
        self.keepMouseCentered = not self.keepMouseCentered
        if self.keepMouseCentered:
            self.setCursor(Qt.BlankCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def load_scene(self):
        fout = open("init_scene.py", "r")
        s = fout.read()
        fout.close()
        d = globals()
        self.context.scene.clear()
        d["context"] = self.context
        for i, l in enumerate(s.split('\n')):
            print(i, l)
        exec(s, d, d)

    def save_scene(self):
        s = "import wol\nimport PyQt5\n\n"

        num = 0
        for o in self.context.scene.children:
            o.save_code_file()
            ss, num = o.serialize(num)
            s += ss
        print(s)
        fout = open("init_scene.py", "w")
        fout.write(s)
        fout.close()

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
                target.on_click(self.context.debug_point, evt)  # Change to on_event
                for b in target.behaviors:
                    b.on_click(self.context.debug_point, evt) # Change to on_event
                if hasattr(target, 'focusable') and target.focusable:
                    if self.context.focused:
                        self.context.focused.focused = False
                        self.context.focused.on_event(Events.LostFocus)
                    self.context.focused = target
                    self.context.focused.on_event(Events.GotFocus)
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
