import numpy
import pybullet
import pywavefront
from OpenGL import GL, GLU
from OpenGL.arrays import vbo
from PyQt5.QtGui import QVector3D, QOpenGLTexture, QImage, QVector4D

from wol import utils
from wol.ShadersLibrary import ShadersLibrary
from wol.SceneNode import SceneNode
import pybullet as pb

class Grid(SceneNode):
    def __init__(self, name="Grid", parent=None):
        SceneNode.__init__(self, name, parent)
        self.vertices = list()

        for i in range(65):
            self.vertices.append(QVector3D(0.0, i - 32, -32.0))
            self.vertices.append(QVector3D(0.0, i - 32, 32.0))
            self.vertices.append(QVector3D(0.0, -32.0, i - 32))
            self.vertices.append(QVector3D(0.0, 32.0, i - 32))

        self.program = None
        self.color = QVector4D(0.2, 0.2, 0.6, 1.0)
        self.layer=2

    def initialize_gl(self):
        #self.program = ShadersLibrary.create_program('simple_color_white')
        self.program = ShadersLibrary.create_program('simple_color')

    def paint(self, program):
        self.program.bind()
        self.program.setAttributeArray(0, self.vertices)
        self.program.setUniformValue('matrix', self.proj_matrix)
        self.program.setUniformValue('material_color', self.color)
        GL.glDrawArrays(GL.GL_LINES, 0, 65*4)
        program.bind()


class WireframeCubeNode(SceneNode):
    def __init__(self, name="WireCube", parent=None, color=QVector4D(0.2, 0.2, 0.6, 1.0)):
        super().__init__(name, parent)
        self.program = None
        self.vertices = list()

        self.vertices.append(QVector3D(-1, -1, -1))
        self.vertices.append(QVector3D(1,  -1, -1))
        self.vertices.append(QVector3D(-1, 1,  -1))
        self.vertices.append(QVector3D(1,  1,  -1))
        self.vertices.append(QVector3D(-1, -1, 1))
        self.vertices.append(QVector3D(1, -1, 1))
        self.vertices.append(QVector3D(-1, 1, 1))
        self.vertices.append(QVector3D(1, 1, 1))

        self.vertices.append(QVector3D(-1, -1, -1))
        self.vertices.append(QVector3D(-1, 1, -1))
        self.vertices.append(QVector3D(1, -1, -1))
        self.vertices.append(QVector3D(1, 1, -1))
        self.vertices.append(QVector3D(-1, -1,  1))
        self.vertices.append(QVector3D(-1, 1,  1))
        self.vertices.append(QVector3D(1, -1,  1))
        self.vertices.append(QVector3D(1, 1,  1))

        self.vertices.append(QVector3D(-1, -1, 1))
        self.vertices.append(QVector3D(-1, -1, -1))
        self.vertices.append(QVector3D(-1, 1, 1))
        self.vertices.append(QVector3D(-1, 1, -1))
        self.vertices.append(QVector3D(1, -1, 1))
        self.vertices.append(QVector3D(1, -1, -1))
        self.vertices.append(QVector3D(1, 1, 1))
        self.vertices.append(QVector3D(1, 1, -1))

        self.color = color
        self.register_collider("cube.urdf")

    def initialize_gl(self):
        self.program = ShadersLibrary.create_program('wireframe')

    def compute_transform(self, project=True):
        super().compute_transform(project)
        pybullet.unsupportedChangeScaling(self.collider_id, (self.scale[0], self.scale[1], self.scale[2]))

    def paint(self, program):
        self.program.bind()
        self.program.setAttributeArray(0, self.vertices)
        self.program.setUniformValue('matrix', self.proj_matrix)
        self.program.setUniformValue('material_color', self.color)
        GL.glDrawArrays(GL.GL_LINES, 0, 24)
        program.bind()


class CubeNode(SceneNode):
    def __init__(self, name="Cube", parent=None, color=QVector4D(0.2, 0.2, 0.6, 1.0)):
        super().__init__(name, parent)
        self.program = None
        self.vertices = list()
        self.normals = list()
        # v1  v2        v5  v6
        #
        # v3  v4        v7  v8

        v1 = QVector3D(-1, 1, -1)
        v2 = QVector3D(1,  1, -1)
        v3 = QVector3D(-1, 1,  1)
        v4 = QVector3D(1,  1,  1)
        v5 = QVector3D(-1, -1, -1)
        v6 = QVector3D(1, -1,  -1)
        v7 = QVector3D(-1, -1,  1)
        v8 = QVector3D(1, -1,   1)

        nup = QVector3D(0., 1., 0.)
        ndown = QVector3D(0., -1., 0.)
        nright = QVector3D(1., 0., 0.)
        nleft = QVector3D(-1., 0., 0.)
        nfront = QVector3D(0., 0., 1.)
        nback = QVector3D(0., 0., -1.)

        # self.vertices += [v1, nup, v2, nup, v3, nup]
        # self.vertices += [v3, nup, v2, nup, v4, nup]
        #
        # self.vertices += [v5, ndown, v6, ndown, v7, ndown]
        # self.vertices += [v7, ndown, v6, ndown, v8, ndown]
        #
        # self.vertices += [v3, nfront, v4, nfront, v7, nfront]
        # self.vertices += [v7, nfront, v4, nfront, v8, nfront]
        #
        # self.vertices += [v1, nback, v2, nback, v5, nback]
        # self.vertices += [v5, nback, v2, nback, v6, nback]
        #
        # self.vertices += [v1, nleft, v5, nleft, v3, nleft]
        # self.vertices += [v3, nleft, v5, nleft, v7, nleft]
        #
        # self.vertices += [v2, nright, v6, nright, v4, nright]
        # self.vertices += [v4, nright, v6, nright, v8, nright]

        self.vertices += [v1, v2, v3, ]
        self.vertices += [v3, v2, v4, ]
        self.vertices += [v5, v6, v7, ]
        self.vertices += [v7, v6, v8, ]
        self.vertices += [v3, v4, v7, ]
        self.vertices += [v7, v4, v8, ]
        self.vertices += [v1, v2, v5, ]
        self.vertices += [v5, v2, v6, ]
        self.vertices += [v1, v5, v3, ]
        self.vertices += [v3, v5, v7, ]
        self.vertices += [v2, v6, v4, ]
        self.vertices += [v4, v6, v8, ]

        self.normals += [nup, nup, nup]
        self.normals += [nup, nup, nup]
        self.normals += [ndown, ndown, ndown]
        self.normals += [ndown, ndown, ndown]
        self.normals += [nfront, nfront, nfront]
        self.normals += [nfront, nfront, nfront]
        self.normals += [nback, nback, nback]
        self.normals += [nback, nback, nback]
        self.normals += [nleft, nleft, nleft]
        self.normals += [nleft, nleft, nleft]
        self.normals += [nright, nright, nright]
        self.normals += [nright, nright, nright]

        self.color = color
        self.register_collider("cube.urdf")

    def initialize_gl(self):
        self.program = ShadersLibrary.create_program('simple_lighting')

    def compute_transform(self, project=True):
        super().compute_transform(project)
        pybullet.unsupportedChangeScaling(self.collider_id, (self.scale[0], self.scale[1], self.scale[2]))

    def paint(self, program):
        self.program.bind()
        self.program.setAttributeArray(0, self.vertices)
        self.program.setAttributeArray(2, self.normals)
        self.program.enableAttributeArray(0)
        self.program.enableAttributeArray(2)

        # self.program.setUniformValue('light_position', QVector3D(1.0, 10.0, -10.0))
        self.program.setUniformValue('light_position', self.context.current_camera.position)
        self.program.setUniformValue('matmodel', self.transform)
        self.program.setUniformValue('material_color', self.color)
        self.program.setUniformValue('mvp', self.proj_matrix)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 36)
        program.bind()


class Sphere(SceneNode):
    def __init__(self, name=None, parent=None):
        SceneNode.__init__(self, name, parent)
        self.quadric = None
        self.size = 1.0
        self.register_collider("sphere.urdf")
        self.program = None
        self.color = QVector4D(0.5, 1.0, 0.5, 1.0)

    def initialize_gl(self):
        self.quadric = GLU.gluNewQuadric()
        self.program = ShadersLibrary.create_program('simple_lighting')
        # self.program = ShadersLibrary.create_program('simple_color')

    def paint(self, program):
        self.program.bind()
        # self.program.setUniformValue('light_position', QVector3D(1.0, 10.0, -10.0))
        self.program.setUniformValue('light_position', self.context.current_camera.position)
        self.program.setUniformValue('matmodel', self.transform)
        self.program.setUniformValue('material_color', self.color)
        self.program.setUniformValue('mvp', self.proj_matrix)
        self.program.setUniformValue('matrix', self.proj_matrix)
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
        self.interpolation = GL.GL_LINEAR
        self.register_collider("plane.urdf")

    def refresh_vertices(self):
        p0 = QVector3D(self.vertices[0][0], self.vertices[0][1], self.vertices[0][2])
        p1 = QVector3D(self.vertices[1][0], self.vertices[1][1], self.vertices[1][2])
        p2 = QVector3D(self.vertices[2][0], self.vertices[2][1], self.vertices[2][2])

    def initialize_gl(self):
        if self.texture_image:
            self.texture = QOpenGLTexture(self.texture_image)

    def paint(self, program):
        program.bind()
        program.setAttributeArray(0, self.vertices)
        program.setAttributeArray(1, self.texCoords)
        if self.texture:
            self.texture.bind()
        program.setUniformValue('matrix', self.proj_matrix)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, self.interpolation)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, self.interpolation)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)


class MeshNode(SceneNode):
    def __init__(self, filename, name="Mesh", parent=None):
        SceneNode.__init__(self, name, parent)
        self.filename = filename
        self.mesh = pywavefront.Wavefront(filename, create_materials=True, collect_faces=True)
        vertices = list()
        for v in self.mesh.vertices:
            vertices += v
        vertices = numpy.array(vertices, dtype='f')
        self.vertices = vbo.VBO(vertices)

        indices = list()
        for ind in self.mesh.mesh_list[0].faces:
            indices += ind
        indices = numpy.array(indices, dtype=numpy.int32)

        self.indices = vbo.VBO(indices, target=GL.GL_ELEMENT_ARRAY_BUFFER)
        self.texture = None

    def paint(self, program):
        program.bind()
        program.setUniformValue('matrix', self.proj_matrix)
        self.indices.bind()
        self.vertices.bind()
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 0, None)

        GL.glDrawElements(GL.GL_TRIANGLES, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        self.indices.unbind()
        self.vertices.unbind()

