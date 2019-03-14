from PyQt5.QtGui import QMatrix4x4, QQuaternion, QVector3D


class SceneNode:
    def __init__(self, name="Node", parent=None):
        self.parent = parent
        if parent:
            self.parent.children.append(self)
            self.context = self.parent.context
        self.children = list()
        self.name = name
        self.position = QVector3D()
        self.orientation = QQuaternion()
        self.transform = QMatrix4x4()
        self.look_at = self.orientation.rotatedVector((QVector3D(1, 0, 0))) + self.position
        self.collider = None
        self.prog_matrix = self.transform

    def compute_transform(self):
        if self.parent:
            m = QMatrix4x4(self.parent.transform)
        else:
            m = QMatrix4x4()
        m.translate(self.position)
        m.rotate(self.orientation)
        self.transform = m
        self.look_at = self.orientation.rotatedVector((QVector3D(1, 0, 0))) + self.position
        self.prog_matrix = self.context.current_camera.projection_matrix * self.transform

    def update(self, dt):
        return

    def update_recurs(self, dt=0.0):
        self.update(dt)
        for c in self.children:
            c.update_recurs(dt)

    def paint(self, program):
        return

    def paint_recurs(self, program):
        self.compute_transform()
        self.paint(program)
        for c in self.children:
            c.paint_recurs(program)

    def initialize_gl(self):
        return

    def initialize_gl_recurs(self):
        self.initialize_gl()
        for c in self.children:
            c.initialize_gl_recurs()

    def add_child(self, child):
        self.children.append(child)
        child.parent = self
        child.context = self.context


class RootNode(SceneNode):
    def __init__(self, context):
        super(RootNode, self).__init__(name="root")
        self.context = context
        # Put that in a separate camera object
        self.position = QVector3D(0.0, 0.0, 0.0)
        self.forward = QVector3D()
        self.up = QVector3D()

    def compute_transform(self):
        m = QMatrix4x4()
        m.translate(self.position)
        m.rotate(self.orientation)
        self.transform = m
        self.look_at = self.orientation.rotatedVector((QVector3D(1, 0, 0))) + self.position


class CameraNode(SceneNode):
    def __init__(self, parent):
        super(CameraNode, self).__init__(name="Camera", parent=parent)
        m = QMatrix4x4()
        m.perspective(30.0, 4.0 / 3.0, 1.0, 100.0)
        self.projection_matrix = m

    def compute_transform(self):
        m = QMatrix4x4()
        m.perspective(30.0, 4.0 / 3.0, 1.0, 100.0)
        m.lookAt(self.position, self.position + self.look_at, QVector3D(0, 1, 0))
        self.projection_matrix = m

