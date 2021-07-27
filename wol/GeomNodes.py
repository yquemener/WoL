import ctypes

import numpy
import odepy
import pybullet
import pywavefront
from OpenGL import GL, GLU
from OpenGL.arrays import vbo
from PyQt5.QtGui import QVector3D, QOpenGLTexture, QImage, QVector4D, QQuaternion

from wol import utils
from wol.Behavior import Behavior
from wol.Constants import UserActions, Events
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
        if init_collider:
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
    def __init__(self, name="Cube", parent=None, color=QVector4D(0.2, 0.2, 0.6, 1.0), init_collider=True):
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
        if init_collider:
            self.register_collider("cube.urdf")

    def initialize_gl(self):
        self.program = ShadersLibrary.create_program('simple_lighting')

    def compute_transform(self, project=True):
        super().compute_transform(project)
        if self.collider_id is not None:
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
    def __init__(self, name=None, parent=None, init_collider=True):
        SceneNode.__init__(self, name, parent)
        self.quadric = None
        self.size = 1.0
        if init_collider:
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
        if init_collider:
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


class UrdfBehavior(Behavior):
    def __init__(self, urdf_id, link_id):
        super().__init__()
        self.urdf_id = urdf_id
        self.link_id = link_id
        self.simulation_enabled = False
        self.enable_at_next_update = False
        self.events_handlers[Events.Ungrabbed].append(self.on_force_position)

    def on_update(self, dt):
        if self.enable_at_next_update:
            self.enable_at_next_update = False
            pb.unsupportedChangeScaling(self.urdf_id, (self.obj.scale[0], self.obj.scale[1], self.obj.scale[2]))
            self.obj.compute_transform()
            wp = self.obj.world_position()
            wo = self.obj.world_orientation()
            # wp = self.obj.position
            # wo = self.obj.orientation
            pb.resetBasePositionAndOrientation(
                self.urdf_id,
                (wp.x(), wp.y(), wp.z()),
                (wo.x(), wo.y(), wo.z(), wo.scalar()))
            self.simulation_enabled = True
            for child in self.obj.children:
                child.sim.simulation_enabled = True

        if self.simulation_enabled:
            if self.link_id == -1:
                pos, orient = pb.getBasePositionAndOrientation(self.urdf_id)
            else:
                pos, orient, *_ = pb.getLinkState(self.urdf_id, self.link_id)

            self.obj.position = QVector3D(pos[0], pos[1], pos[2])
            self.obj.orientation = QQuaternion(orient[3], orient[0], orient[1], orient[2])

    def set_simulation(self, enabled):
        if enabled:
            self.enable_at_next_update = True

    def on_force_position(self):
        print("t")
        wp = self.obj.world_position()
        wo = self.obj.world_orientation()
        print(wp)
        pb.resetBasePositionAndOrientation(
            self.urdf_id, (wp.x(), wp.y(), wp.z()),
            (wo.x(), wo.y(), wo.z(), wo.scalar()))


class UrdfSingleNode(SceneNode):
    def __init__(self, filename, name="urdf", parent=None, static=False):
        SceneNode.__init__(self, name, parent)
        if static:
            fixed_base = 1
        else:
            fixed_base = 0

        self.urdf_id = pb.loadURDF(filename, useFixedBase=fixed_base)
        self.properties["skip serialization"] = True
        self.add_behavior(UrdfBehavior(self.urdf_id, -1))


class UrdfNode(SceneNode):
    def __init__(self, filename, name="urdf", parent=None, static=False):
        SceneNode.__init__(self, name, parent)
        if static:
            fixed_base = 1
        else:
            fixed_base = 0

        self.urdf_id = pb.loadURDF(filename, useFixedBase=fixed_base)
        self.context.bullet_ids[self.urdf_id] = self
        self.initialized_children = False
        self.sim = UrdfBehavior(self.urdf_id, -1)
        self.add_behavior(self.sim)
        self.properties["skip serialization"] = True

    def update(self, dt):
        if not self.initialized_children:
            self.initialized_children = True

            visual_data = pb.getVisualShapeData(self.urdf_id)
            nJoints = pb.getNumJoints(self.urdf_id)
            for i in range(nJoints):
                # info = pb.getJointInfo(self.collider_id, i)
                # print(info)
                obj_id, link_id, shape, scale, _, _, _, _ = visual_data[i]
                name = f"{self.name}_element_{i}"
                if shape == 4:
                    child = Sphere(name, init_collider=False)
                elif shape == 3:
                    child = CubeNode(name, color=QVector4D(0.2,0.7, .2, 1.0), init_collider=False)
                else:
                    print(f"Unknown shape: {shape}")
                child.simulation_enabled = False
                child.scale = QVector3D(scale[0], scale[1], scale[2])
                # print(obj_id, link_id)
                if link_id == -1:
                    pos, orient = pb.getBasePositionAndOrientation(obj_id)
                else:
                    pos, orient, *_ = pb.getLinkState(obj_id, link_id)
                    print(link_id)
                # print(obj_id, link_id, pos)
                child.position = QVector3D(pos[0], pos[1], pos[2])
                child.orientation = QQuaternion(orient[0], orient[1], orient[2], orient[3])
                behav = UrdfBehavior(obj_id, link_id)
                child.add_behavior(behav)
                child.sim = behav
                child.reparent(self)
                # self.add_child(child)
                self.context.bullet_ids[obj_id] = self

    #         # pb.getVisualShapeData(self.collider_id)
    #         # objectUniqueId
    #         # linkIndex
    #         # visualGeometryType : 3=cylinder, 4=box ...
    #         # dimensions
    #         # meshAssetFileName
    #         # localVisualFrame position
    #         # localVisualFrame orientation
    #         # rgbaColor
    #         # textureUniqueId
    #
    #         shapeData = pb.getVisualShapeData(self.collider_id)
    #         for i in range(nJoints):
    #             info = pb.getJointInfo(self.collider_id, i)
    #             print(info)
    #             print(pb.getBasePositionAndOrientation(info[0]))
    #
    #         pb.unsupportedChangeScaling(self.collider_id, (self.wscale, self.hscale, 1.0))
    #         pb.resetBasePositionAndOrientation(
    #             self.collider_id, (wp.x(), wp.y(), wp.z()),
    #             (wo.x(), wo.y(), wo.z(), wo.scalar()))


class OdeBehavior(Behavior):
    def __init__(self, body, geom, obj=None):
        super(OdeBehavior, self).__init__(obj=obj)
        self.body = body
        self.geom = geom
        odepy.dGeomSetBody(self.geom, self.body)
        odepy.dBodySetPosition(self.body, self.obj.position[0], self.obj.position[1], self.obj.position[2])

    def on_update(self, dt):
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
