from OpenGL import GL, GLU
from PyQt5.QtGui import QVector3D, QOpenGLTexture, QImage

from wol import Collisions, utils
from wol.ShadersLibrary import ShadersLibrary
from wol.SceneNode import SceneNode


class Grid(SceneNode):
    def __init__(self, name="Grid", parent=None):
        SceneNode.__init__(self, name, parent)
        self.vertices = list()

        for i in range(65):
            self.vertices.append(QVector3D(0.0, i - 32, -32.0))
            self.vertices.append(QVector3D(0.0, i - 32, 32.0))
            self.vertices.append(QVector3D(0.0, -32.0, i - 32))
            self.vertices.append(QVector3D(0.0, 32.0, i - 32))

        self.collider = Collisions.Plane()
        self.program = None

    def initialize_gl(self):
        self.program = ShadersLibrary.create_program('simple_color_white')

    def paint(self, program):
        self.program.bind()
        self.program.setAttributeArray(0, self.vertices)
        self.program.setUniformValue('matrix', self.prog_matrix)
        GL.glDrawArrays(GL.GL_LINES, 0, 65*4)
        program.bind()


class Sphere(SceneNode):
    def __init__(self, name=None, parent=None):
        SceneNode.__init__(self, name, parent)
        self.quadric = None
        self.size = 1.0
        self.collider = Collisions.Sphere()

    def initialize_gl(self):
        self.quadric = GLU.gluNewQuadric()
        self.program = ShadersLibrary.create_program('simple_color_white')

    def paint(self, program):
        self.program.bind()
        self.program.setUniformValue('matrix', self.prog_matrix)
        GLU.gluSphere(self.quadric, self.size, 20, 20)
        program.bind()


class CardNode(SceneNode):
    def __init__(self, filename=None, name="Card", parent=None):
        SceneNode.__init__(self, name, parent)
        self.filename = filename
        if filename:
            self.texture_image = QImage(filename)
        else:
            self.texture_image = None
        self.texture = None

        self.vertices = utils.generate_square_vertices_fan()
        self.texCoords = utils.generate_square_texcoords_fan()
        self.refresh_vertices()

    def refresh_vertices(self):
        p0 = QVector3D(self.vertices[0][0], self.vertices[0][1], self.vertices[0][2])
        p1 = QVector3D(self.vertices[1][0], self.vertices[1][1], self.vertices[1][2])
        p2 = QVector3D(self.vertices[2][0], self.vertices[2][1], self.vertices[2][2])
        self.collider = Collisions.Outline3D(normale=QVector3D.crossProduct(p2-p0, p1-p0).normalized())
        for v in self.vertices:
            self.collider.add_3d_point(QVector3D(*v))

    def initialize_gl(self):
        if self.texture_image:
            self.texture = QOpenGLTexture(self.texture_image)

    def paint(self, program):
        program.bind()
        program.setAttributeArray(0, self.vertices)
        program.setAttributeArray(1, self.texCoords)
        if self.texture:
            self.texture.bind()
        program.setUniformValue('matrix', self.prog_matrix)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE);
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE);
        GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)

