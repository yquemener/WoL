from OpenGL import GL
from PyQt5.QtGui import QMatrix4x4, QQuaternion, QVector3D, QImage, QOpenGLTexture
import struct

from wol import utils


class SceneNode:
    next_uid = 0
    uid_map = dict()

    @staticmethod
    def new_uid():
        SceneNode.next_uid += 1
        return SceneNode.next_uid

    def __init__(self, name="Node", parent=None):
        self.parent = parent
        if parent:
            self.parent.children.append(self)
            self.context = self.parent.context
        self.set_uid(SceneNode.new_uid())
        self.children = list()
        self.name = name
        self.position = QVector3D()
        self.scale = QVector3D(1, 1, 1)
        self.orientation = QQuaternion()
        self.transform = QMatrix4x4()
        self.look_at = self.orientation.rotatedVector((QVector3D(1, 0, 0))) + self.position
        self.collider = None
        self.prog_matrix = self.transform
        self.visible = True
        self.properties = dict()
        """ Layers:
            0: skybox
            1: regular objects
            2: transparent objects
            3: HUD UI
            """
        self.layer = 1

    def set_uid(self, uid):
        self.uid = uid
        SceneNode.next_uid = max(SceneNode.next_uid, uid+1)
        SceneNode.uid_map[uid] = self

    def reparent(self, new_parent, transform=True):
        if transform:
            new_transform = new_parent.transform.inverted()[0] * self.transform
            self.position = new_transform.map(QVector3D())
            self.orientation = QQuaternion.fromDirection(new_transform.mapVector(QVector3D(0, 0, 1)),
                                                         new_transform.mapVector(QVector3D(0, 1, 0)))
        if self.parent:
            self.parent.children.remove(self)
            self.parent = new_parent
            self.parent.children.append(self)

    def compute_transform(self):
        if self.parent:
            m = QMatrix4x4(self.parent.transform)
        else:
            m = QMatrix4x4()
        m.translate(self.position)
        m.rotate(self.orientation)
        m.scale(self.scale)
        self.transform = m
        self.look_at = self.orientation.rotatedVector((QVector3D(1, 0, 0))) + self.position
        self.prog_matrix = self.context.current_camera.projection_matrix * self.transform
        if self.collider is not None:
            self.collider.transform = self.transform

    def world_position(self):
        if self.parent:
            return self.parent.transform.map(self.position)
        else:
            return self.position

    def update(self, dt):
        return

    def update_recurs(self, dt=0.0):
        self.update(dt)
        self.compute_transform()
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

    def serialize_recurs(self):
        s = self.serialize()
        for c in self.children:
            s += c.serialize_recurs()
        return s

    def make_pose_packet(self):
        return struct.pack("!10d",
                           self.position.x(),
                           self.position.y(),
                           self.position.z(),
                           self.orientation.scalar(),
                           self.orientation.x(),
                           self.orientation.y(),
                           self.orientation.z(),
                           self.scale.x(),
                           self.scale.y(),
                           self.scale.z())

    def update_from_pose_packet(self, packet):
        p = struct.unpack("!10d", packet)
        self.position = QVector3D(p[0], p[1], p[2])
        self.orientation = QQuaternion(p[3], p[4], p[5], p[6])
        self.scale = QVector3D(p[7], p[8], p[9])

    def serialize(self):
        s = "new " + str(self.__class__.__name__) + "\n"
        if hasattr(self, "filename") and self.filename is not None:
            s += "filename=" + str(self.filename) + "\n"
        if hasattr(self, "name"):
            s += "name=" + str(self.name) + "\n\n"

        if hasattr(self, "position"):
            s += "move " + str(self.position.x()) + " " \
                 + str(self.position.y()) + " " \
                 + str(self.position.z()) + "\n"
        if hasattr(self, "scale"):
            s += "scale " + str(self.scale.x()) + " " \
                 + str(self.scale.y()) + " " \
                 + str(self.scale.z()) + "\n"
        if hasattr(self, "orientation"):
            s += "rotate " + str(self.orientation.scalar()) + " " \
                 + str(self.orientation.x()) + " " \
                 + str(self.orientation.y()) + " " \
                 + str(self.orientation.z()) + "\n"""
        s += "setuid " + str(self.uid) + "\n"
        s += "setparentuid " + str(self.parent.uid) + "\n"
        s += "\n"
        return s

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
    def __init__(self, context, name="root"):
        super(RootNode, self).__init__(name=name)
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
    def __init__(self, parent, name):
        super(CameraNode, self).__init__(name="Camera", parent=parent)
        self.angle = 30.0
        self.ratio = 4.0/3.0
        self.projection_matrix = QMatrix4x4()

    def compute_transform(self):
        m = QMatrix4x4()
        m.rotate(self.orientation)
        la = self.position + m.mapVector(QVector3D(0, 0, 1))

        m = QMatrix4x4()
        m.perspective(self.angle, self.ratio, 1.0, 1000.0)
        m.lookAt(self.position, la, QVector3D(0, 1, 0))
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
