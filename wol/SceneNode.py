from OpenGL import GL
from PyQt5.QtGui import QMatrix4x4, QQuaternion, QVector3D, QImage, QOpenGLTexture

from wol import utils


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
        self.visible = True
        """ Layers:
            0: skybox
            1: regular objects
            2: transparent objects
            3: HUD UI
            """
        self.layer = 1

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
        if self.collider is not None:
            self.collider.transform = self.transform

    def update(self, dt):
        return

    def update_recurs(self, dt=0.0):
        self.update(dt)
        for c in self.children:
            c.update_recurs(dt)

    def paint(self, program):
        return

    def paint_recurs(self, program, layer=1):
        self.compute_transform()
        if self.visible:
            if self.layer == layer:
                self.paint(program)

        for c in self.children:
            c.paint_recurs(program, layer)

    def initialize_gl(self):
        return

    def initialize_gl_recurs(self):
        self.initialize_gl()
        for c in self.children:
            c.initialize_gl_recurs()

    def collide_recurs(self, ray):
        collisions = list()
        if self.collider is not None and self.visible:
            r, cc = self.collider.collide_with_ray(ray)
            if r:
                collisions.append((self, cc))
        for c in self.children:
            collisions += c.collide_recurs(ray)
        return collisions

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
        self.angle = 30.0
        self.ratio = 4.0/3.0
        self.projection_matrix = QMatrix4x4()

    def compute_transform(self):
        m = QMatrix4x4()
        m.perspective(self.angle, self.ratio, 1.0, 1000.0)
        m.lookAt(self.position, self.position + self.look_at, QVector3D(0, 1, 0))
        self.projection_matrix = m

        if self.parent:
            m = QMatrix4x4(self.parent.transform)
        else:
            m = QMatrix4x4()
        m.translate(self.position)
        m.rotate(self.orientation)
        self.transform = m


class SkyBox(SceneNode):
    def __init__(self, parent,
                 texture_filenames=("resources/cubemaps/bkg/lightblue/back.png",
                                    "resources/cubemaps/bkg/lightblue/front.png",
                                    "resources/cubemaps/bkg/lightblue/left.png",
                                    "resources/cubemaps/bkg/lightblue/right.png",
                                    "resources/cubemaps/bkg/lightblue/top.png",
                                    "resources/cubemaps/bkg/lightblue/bot.png"),
                 name="SkyBox"):
        super(SkyBox, self).__init__(parent=parent, name=name)
        self.layer = 0
        self.textures = list()
        self.texture_images = list()
        for fn in texture_filenames:
            self.texture_images.append(QImage(fn))
        self.face_verts = utils.generate_square_vertices_fan()
        self.face_uvs = utils.generate_square_texcoords_fan()
        self.face_transforms = list()
        for euler in ((+ 0, 180, 0),
                      (+ 0,   0, 0),
                      (+ 0,  90, 0),
                      (+ 0, -90, 0),
                      (-90,   0, 0),
                      (+90,   0, 0)):
            m = QMatrix4x4()
            m.rotate(QQuaternion.fromEulerAngles(*euler))
            m.translate(0, 0, 10)
            m.scale(10.0)
            self.face_transforms.append(m)

    def initialize_gl(self):
        for img in self.texture_images:
            self.textures.append(QOpenGLTexture(img))

    def compute_transform(self):
        if self.parent:
            m = QMatrix4x4(self.parent.transform)
        else:
            m = QMatrix4x4()
        # Remove rotation from parent transform:
        translate_vec = m.map(QVector3D())
        m = QMatrix4x4()
        m.translate(translate_vec)
        m.translate(self.position)
        m.rotate(self.orientation)
        self.transform = m
        self.look_at = self.orientation.rotatedVector((QVector3D(1, 0, 0))) + self.position
        self.prog_matrix = self.context.current_camera.projection_matrix * self.transform
        if self.collider is not None:
            self.collider.transform = self.transform

    def paint(self, program):
        program.bind()
        program.setAttributeArray(0, self.face_verts)
        program.setAttributeArray(1, self.face_uvs)
        GL.glPushAttrib(GL.GL_DEPTH_WRITEMASK)
        GL.glDepthMask(False)
        for i in range(6):
            self.textures[i].bind()
            program.setUniformValue('matrix', self.prog_matrix * self.face_transforms[i])
            GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)
        GL.glPopAttrib(GL.GL_DEPTH_WRITEMASK)
