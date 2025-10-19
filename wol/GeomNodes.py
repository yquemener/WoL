import ctypes

import numpy
import odepy
import pywavefront
from OpenGL import GL, GLU
from OpenGL.arrays import vbo
from PyQt5.QtGui import QVector3D, QOpenGLTexture, QImage, QVector4D, QQuaternion

from wol import utils
from wol.Behavior import Behavior
from wol.Constants import UserActions, Events
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

        self.program = None
        self.color = QVector4D(0.2, 0.2, 0.6, 1.0)
        # self.layer=2

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
    def __init__(self, name="WireCube", parent=None, color=QVector4D(0.2, 0.2, 0.6, 1.0), init_collider=True):
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

    def initialize_gl(self):
        self.program = ShadersLibrary.create_program('wireframe')

    def compute_transform(self, project=True):
        super().compute_transform(project)

    def paint(self, program):
        self.program.bind()
        self.program.setAttributeArray(0, self.vertices)
        self.program.setUniformValue('matrix', self.proj_matrix)
        self.program.setUniformValue('material_color', self.color)
        GL.glDrawArrays(GL.GL_LINES, 0, 24)
        program.bind()


class CubeNode(SceneNode):
    def __init__(self, name="Cube", parent=None, color=QVector4D(0.2, 0.2, 0.6, 1.0), init_collider=True):
        super().__init__(name, parent)
        self.program = None
        self.vertices = utils.generate_cube_vertices()
        self.normals = list()

        nup = QVector3D(0., 1., 0.)
        ndown = QVector3D(0., -1., 0.)
        nright = QVector3D(1., 0., 0.)
        nleft = QVector3D(-1., 0., 0.)
        nfront = QVector3D(0., 0., 1.)
        nback = QVector3D(0., 0., -1.)

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
        if init_collider:
            self.ode = OdeBoxBehavior(obj=self, kinematic=True)
            self.add_behavior(self.ode)

    def initialize_gl(self):
        self.program = ShadersLibrary.create_program('simple_lighting')

    def compute_transform(self, project=True):
        super().compute_transform(project)

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
    def __init__(self, name=None, parent=None, init_collider=True):
        SceneNode.__init__(self, name, parent)
        self.quadric = None
        self.size = 1.0
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

class Avatar(SceneNode):
    def __init__(self, name=None, parent=None, init_collider=True):
        SceneNode.__init__(self, name, parent)
        big_sphere = Sphere(name=name+"_bs", parent=self)
        small_sphere = Sphere(name=name + "_ss", parent=self)
        small_sphere.position = QVector3D(0,0,1)
        small_sphere.scale = QVector3D(0.2,0.2,0.2)
        # self.add_child()


class CardNode(SceneNode):
    def __init__(self, filename=None, name="Card", parent=None, init_collider=True):
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
        # self.vertices = vertices

        normals = list()
        for n in self.mesh.parser.normals:
            normals += n
        normals = numpy.array(normals, dtype='f')
        self.normals = vbo.VBO(normals)
        # self.normals = normals

        indices = list()
        for ind in self.mesh.mesh_list[0].faces:
            indices += ind
        indices = numpy.array(indices, dtype=numpy.int32)
        self.indices = vbo.VBO(indices, target=GL.GL_ELEMENT_ARRAY_BUFFER)
        # self.indices = indices

        self.texture = None

        self.color = QVector4D(0.0, 0.0, 1.0, 1.0)

    def initialize_gl(self):
        self.program = ShadersLibrary.create_program('simple_lighting')
        # self.program = ShadersLibrary.create_program('simple_texture')

    def paint(self, program):
        self.program.bind()
        self.program.setUniformValue('light_position', self.context.current_camera.position)
        self.program.setUniformValue('matmodel', self.transform)
        self.program.setUniformValue('material_color', self.color)
        self.program.setUniformValue('mvp', self.proj_matrix)
        self.program.setUniformValue('matrix', self.proj_matrix)
        self.indices.bind()
        GL.glEnableVertexAttribArray(0)
        self.vertices.bind()
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 0, None)

        GL.glEnableVertexAttribArray(2)
        # GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, 0, None)
        self.normals.bind()
        GL.glVertexAttribPointer(2, 3, GL.GL_FLOAT, False, 0, None)


        # program.setAttributeArray(0, self.mesh.vertices)
        # program.setAttributeArray(1, self.mesh.vertices)
        # program.setAttributeArray(2, self.mesh.parser.normals)

        GL.glDrawElements(GL.GL_TRIANGLES, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        self.indices.unbind()
        self.vertices.unbind()
        self.normals.unbind()


class OdeBehavior(Behavior):
    def __init__(self, body, geom, obj=None, kinematic=False):
        super(OdeBehavior, self).__init__(obj=obj)
        self.body = body
        self.geom = geom
        self.offset = QVector3D(0., 0., 0.)
        obj.context.register_collider(self.geom, self.obj)
        odepy.dGeomSetBody(self.geom, self.body)
        wp = self.obj.world_position()
        odepy.dBodySetPosition(self.body, wp[0], wp[1], wp[2])
        wo = self.obj.world_orientation()
        q = odepy.dQuaternion()
        q[0] = wo.x()
        q[1] = wo.y()
        q[2] = wo.z()
        q[3] = wo.scalar()
        odepy.dBodySetQuaternion(self.body, q)
        odepy.dGeomSetQuaternion(self.geom, q)
        if kinematic:
            odepy.dBodySetKinematic(self.body)

    def set_kinematic(self, val):
        if val:
            odepy.dBodySetKinematic(self.body)
        else:
            odepy.dBodySetDynamic(self.body)

    def on_update(self, dt):
        if odepy.dBodyIsKinematic(self.body):
            wp = self.obj.world_position() + self.offset
            odepy.dBodySetPosition(self.body, wp[0], wp[1], wp[2])
            odepy.dGeomSetPosition(self.geom, wp[0], wp[1], wp[2])
            wo = self.obj.world_orientation()
            # wo = self.obj.parent.orientation
            q = odepy.dQuaternion()
            q[0] = wo.scalar()
            q[1] = wo.x()
            q[2] = wo.y()
            q[3] = wo.z()
            odepy.dBodySetQuaternion(self.body, q)
            odepy.dGeomSetQuaternion(self.geom, q)
        else:
            self.obj.position = QVector3D(*odepy.dBodyGetPosition(self.body)[:3])


class OdeSphereBehavior(OdeBehavior):
    def __init__(self, weight,  obj):
        self.weight = weight
        self.radius = obj.size
        mass = odepy.dMass()
        odepy.dMassSetZero(ctypes.byref(mass))
        odepy.dMassSetSphereTotal(ctypes.byref(mass), weight, self.radius)
        body = odepy.dBodyCreate(obj.context.ode_world)
        odepy.dBodySetMass(body, ctypes.byref(mass))
        geom = odepy.dCreateSphere(obj.context.ode_space, self.radius)
        odepy.dGeomSetBody(geom, body)
        odepy.dBodySetPosition(body, obj.position[0], obj.position[1], obj.position[2])
        super(OdeSphereBehavior, self).__init__(body, geom, obj=obj)


class OdeRayBehavior(Behavior):
    def __init__(self, obj, length=1000):
        super(OdeRayBehavior, self).__init__(obj=obj)
        self.length=length
        geom = odepy.dCreateRay(obj.context.ode_space, length)
        self.geom = geom

    def set_ray(self, pos, dirv):
        odepy.dGeomRaySet(self.geom, pos[0], pos[1], pos[2], dirv[0], dirv[1], dirv[2])
        odepy.dGeomRaySetLength(self.geom, self.length)

    def on_update(self, dt):
        return


class OdeBoxBehavior(OdeBehavior):
    def __init__(self, obj, weight=1, kinematic=False):
        self.weight = weight
        geom = odepy.dCreateBox(obj.context.ode_space, obj.scale.x()*2, obj.scale.y()*2, obj.scale.z()*2)
        body = odepy.dBodyCreate(obj.context.ode_world)
        self.mass = mass = odepy.dMass()
        odepy.dMassSetZero(ctypes.byref(mass))
        odepy.dMassSetBox(ctypes.byref(mass), weight, obj.scale.x()*2, obj.scale.y()*2, obj.scale.z()*2)
        odepy.dBodySetMass(body, ctypes.byref(mass))
        super(OdeBoxBehavior, self).__init__(body, geom, obj=obj, kinematic=kinematic)
        self.obj.events_handlers[Events.GeometryChanged].append(self.on_geometry_changed)
        # print(f"C {self.obj.scale.x()}")

    def on_geometry_changed(self):
        odepy.dMassSetBox(ctypes.byref(self.mass), self.weight, self.obj.scale.x()*2, self.obj.scale.y()*2, self.obj.scale.z()*2)
        wp = self.obj.world_position()
        odepy.dGeomBoxSetLengths(self.geom, self.obj.scale.x()*2, self.obj.scale.y()*2, self.obj.scale.z()*2)
        # print(f"CG {self.obj.scale.x()}")


class OdeTextBehavior(OdeBehavior):
    def __init__(self, obj, weight=1, kinematic=False):
        self.weight = weight
        mult = 2.
        geom = odepy.dCreateBox(obj.context.ode_space, obj.scale.x() * mult, obj.scale.y() * mult, obj.scale.z() * mult)
        # geom = odepy.dCreateBox(obj.context.ode_space, 0.01, 0.01, 0.01)
        body = odepy.dBodyCreate(obj.context.ode_world)
        self.mass = mass = odepy.dMass()
        odepy.dMassSetZero(ctypes.byref(mass))
        odepy.dMassSetBox(ctypes.byref(mass), weight, obj.scale.x() * mult, obj.scale.y() * mult, obj.scale.z() * mult)
        # odepy.dMassSetBox(ctypes.byref(mass), weight, 0.001, 0.001, 0.001)
        odepy.dBodySetMass(body, ctypes.byref(mass))
        super(OdeTextBehavior, self).__init__(body, geom, obj=obj, kinematic=kinematic)
        self.obj.events_handlers[Events.GeometryChanged].append(self.on_geometry_changed)
        print(f"C {self.obj.scale.x()} {self.obj.scale.y()}")

    def on_geometry_changed(self):
        mult = 2.
        odepy.dMassSetBox(ctypes.byref(self.mass), self.weight, self.obj.scale.x()*mult, self.obj.scale.y()*mult, self.obj.scale.z()*mult)
        wp = self.obj.world_position()
        odepy.dGeomBoxSetLengths(self.geom, self.obj.scale.x()*mult, self.obj.scale.y()*mult, self.obj.scale.z()*mult)
        print(f"CG4 {self.obj.scale.x()} {self.obj.scale.y()}")
