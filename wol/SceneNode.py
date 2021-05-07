import traceback
from collections import defaultdict

from OpenGL import GL
from PyQt5.QtGui import QMatrix4x4, QQuaternion, QVector3D, QImage, QOpenGLTexture, QTransform, QMatrix3x3, QVector4D
import struct
import inspect
from threading import Lock
from wol import utils


def instanciate_from_project_file(filename, class_name, args):
    code = open(filename, "r").read()
    exec(code, globals(), globals())
    cl = globals()[class_name](**args)
    cl.source_file = filename
    cl.code = code
    return cl


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
        self.proj_matrix = self.transform
        self.visible = True
        self.properties = dict()
        self.behaviors = list()
        self.tooltip = None
        """ Layers:
            0: skybox
            1: regular objects
            2: transparent objects
           -1: HUD UI
            """
        self.layer = 1
        self.gl_initialized = False
        self.events_handlers = defaultdict(list)
        self.source_file = None
        self.code = None
        self.error_window = None
        self.serialization_atributes = ["position", "orientation", "scale", "properties", "color", "visible"]

    def set_uid(self, uid):
        self.uid = uid
        SceneNode.next_uid = max(SceneNode.next_uid, uid + 1)
        SceneNode.uid_map[uid] = self

    def reparent(self, new_parent, transform=True):
        if transform:
            new_transform = new_parent.transform.inverted()[0] * self.transform
            self.position = new_transform.map(QVector3D())
            self.orientation = QQuaternion.fromDirection(new_transform.mapVector(QVector3D(0, 0, 1)),
                                                         new_transform.mapVector(QVector3D(0, 1, 0)))
        if self.parent:
            try:    # If object has been removed, this will fail
                self.parent.children.remove(self)
            except:
                pass
            self.parent = new_parent
            self.parent.children.append(self)

    def on_event(self, action):
        for h in self.events_handlers[action]:
            h()
        for b in self.behaviors:
            for h in b.events_handlers[action]:
                h()
        for c in self.children:
            c.on_event(action)

    def compute_transform(self, project=True):
        if self.parent:
            m = QMatrix4x4(self.parent.transform)
        else:
            m = QMatrix4x4()
        m.translate(self.position)
        m.rotate(self.orientation)
        m.scale(self.scale)
        self.transform = m
        self.look_at = self.orientation.rotatedVector((QVector3D(1, 0, 0))) + self.position
        if project:
            self.proj_matrix = self.context.current_camera.projection_matrix * self.transform
        else:
            self.proj_matrix = self.transform
        if self.collider is not None:
            self.collider.transform = self.transform

    def world_position(self):
        if self.parent:
            return QVector3D(self.transform.map(QVector4D(0, 0, 0, 1)))
        else:
            return self.position

    def world_orientation(self):
        if self.parent:
            melements = self.transform.data()
            m3 = QMatrix3x3((melements[0], melements[1], melements[2],
                             melements[4], melements[5], melements[6],
                             melements[8], melements[9], melements[10])).transposed()
            return QQuaternion.fromRotationMatrix(m3)
        else:
            return self.orientation

    def set_world_orientation(self, new_orient):
        if self.parent:
            t = self.parent.transform.inverted()[0]

            new_front = new_orient.rotatedVector(QVector3D(0, 0, 1))
            new_up = new_orient.rotatedVector(QVector3D(0, 1, 0))
            self.orientation = QQuaternion.fromDirection(t.mapVector(new_front),
                                                         t.mapVector(new_up))

            # m = self.parent.transform.inverted()[0]
            # me = new_orient.toRotationMatrix().data()
            # m *= QMatrix4x4(me[0], me[1], me[2], 0.,
            #                 me[3], me[4], me[5], 0.,
            #                 me[6], me[7], me[8], 0.,
            #                 0., 0., 0., 1.)
            # md = m.data()
            # m3 = QMatrix3x3((md[0], md[1], md[2],
            #                  md[4], md[5], md[6],
            #                  md[8], md[9], md[10]))
            # self.orientation = QQuaternion.fromRotationMatrix(m3)
        else:
            self.orientation = new_orient

    def set_world_position(self, new_pos):
        if self.parent:
            self.position = QVector3D(self.parent.transform.inverted()[0].map(QVector4D(new_pos, 1)))
        else:
            self.position = new_pos

    def update(self, dt):
        return

    def remove(self):
        # TODO: Check if this is enough to make the garbage collector delete the object
        for c in self.children:
            c.remove()
        self.parent.children.remove(self)

    def show_error(self, e):
        traceback.print_exc()
        # message = str(e)
        # if self.error_window is None:
        #     from wol.TextEditNode import ErrorWindow
        #     self.error_window = ErrorWindow(parent=self, text=message)
        # if self.error_window.text != message:
        #     self.error_window.set_text(message)
        # self.error_window.visible = True

    def hide_error(self):
        if self.error_window is not None:
            self.error_window.visible = False

    def update_recurs(self, dt=0.0):
        try:
            self.update(dt)
        except Exception as e:
            self.show_error(e)
        self.compute_transform()
        behaviors_to_remove = list()
        for b in self.behaviors:
            try:
                b.on_update(dt)
            except Exception as e:
                self.show_error(e)
            if b.kill_me:
                behaviors_to_remove.append(b)
        for b in behaviors_to_remove:
            self.behaviors.remove(b)
        for c in self.children:
            c.update_recurs(dt)

    def paint(self, program):
        return

    def paint_recurs(self, program, layer=1):
        self.compute_transform(project=(layer != -1))
        if self.visible:
            if self.layer == layer:
                try:
                    if not self.gl_initialized:
                        self.initialize_gl()
                    self.paint(program)
                except Exception as e:
                    self.show_error(e)
                else:
                    self.hide_error()

        for c in self.children:
            c.paint_recurs(program, layer)

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

    def save_code_file(self):
        if self.code is not None and len(self.code) > 0:
            if self.source_file is not None:
                f = open(self.source_file, "w")
                f.write(self.code)
                f.close()

    def serialize(self, current_obj_num):
        exclude_types = ["wol.PythonFileEditorNode.InstanciatedObject",
                         "wol.SceneNode.CameraNode"]

        s = "\n"
        if not self.properties.get("skip serialization", False):
            if self.__class__.__module__ == "__main__":
                return "", current_obj_num
            if f"{self.__class__.__module__}.{self.__class__.__name__}" in exclude_types:
                return "", current_obj_num

            if self.source_file is None:
                s += f"obj_{current_obj_num} = {self.__class__.__module__}.{self.__class__.__name__}("
            else:
                s += f"obj_{current_obj_num} = wol.SceneNode.instanciate_from_project_file('{self.source_file}',"
                s += f"'{self.__class__.__name__.split('.')[-1]}', {{"
            if self.source_file is None:
                s += "parent=context.scene"
            else:
                s += "'parent':context.scene"
            constructor_args = list(inspect.signature(self.__init__).parameters)
            for arg in constructor_args:
                if arg == "parent":
                    continue
                if hasattr(self, arg):
                    val = getattr(self, arg)
                    if val is not None:
                        if type(val) is str:
                            rendered = '"' + val.replace("\n", "\\n") + '"'
                        else:
                            rendered = str(val)
                        if self.source_file is None:
                            s += f",{arg}={rendered}"
                        else:
                            s += f",'{arg}':{rendered}"
            if self.source_file is None:
                s += ")\n"
            else:
                s += "})\n"

            for att in self.serialization_atributes:
                if hasattr(self, att):
                    s += f"obj_{current_obj_num}.{att} = {repr(getattr(self, att))}\n"
        return s, current_obj_num+1

    def initialize_gl(self):
        return

    def initialize_gl_recurs(self):
        self.initialize_gl()
        self.gl_initialized = True
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

    def add_behavior(self, behavior):
        behavior.obj = self
        self.behaviors.append(behavior)
        return behavior

    def get_behavior(self, class_name):
        for b in self.behaviors:
            if b.__class__.__name__ == class_name:
                return b
        return None

    def get_behaviors(self, class_name):
        ret_list = list()
        for b in self.behaviors:
            if b.__class__.__name__ == class_name:
                ret_list.append(b)
        return ret_list

    def on_click(self, pos, evt):
        return


class RootNode(SceneNode):
    def __init__(self, context, name="root"):
        super(RootNode, self).__init__(name=name)
        self.context = context
        # Put that in a separate camera object
        self.position = QVector3D(0.0, 0.0, 0.0)
        self.forward = QVector3D()
        self.up = QVector3D()
        self.lock = Lock()

    def clear(self):
        # TODO: actually clear everything cleanly and recursively
        # for c in self.children:
        #     del c
        for c in self.children:
            print(c)
        self.children = list()
        self.children.append(self.context.current_camera)


class CameraNode(SceneNode):
    def __init__(self, parent, name):
        super(CameraNode, self).__init__(name="Camera", parent=parent)
        self.angle = 45.0
        self.ratio = 4.0 / 3.0
        self.projection_matrix = QMatrix4x4()

    def compute_transform(self, project=True):
        m = QMatrix4x4()
        m.rotate(self.orientation)
        la = self.position + m.mapVector(QVector3D(0, 0, 1))

        m = QMatrix4x4()
        m.perspective(self.angle/self.ratio, self.ratio, 1.0, 1000.0)
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
        self.layer = 3
        self.textures = list()
        self.texture_images = list()
        for fn in texture_filenames:
            self.texture_images.append(QImage(fn))
        self.face_verts = utils.generate_square_vertices_fan()
        self.face_uvs = [
            [0.999, 0.999],
            [0.001, 0.999],
            [0.001, 0.001],
            [0.999, 0.001]]
        self.face_transforms = list()
        for euler in ((+ 0, 180, 0),
                      (+ 0, 0, 0),
                      (+ 0, 90, 0),
                      (+ 0, -90, 0),
                      (-90, 0, 0),
                      (+90, 0, 0)):
            m = QMatrix4x4()
            m.rotate(QQuaternion.fromEulerAngles(*euler))
            m.translate(0, 0, 10)
            m.scale(10.0)
            self.face_transforms.append(m)

    def initialize_gl(self):
        for img in self.texture_images:
            self.textures.append(QOpenGLTexture(img))

    def compute_transform(self, project=True):
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
        self.proj_matrix = self.context.current_camera.projection_matrix * self.transform
        if self.collider is not None:
            self.collider.transform = self.transform

    def paint(self, program):
        program.bind()
        program.setAttributeArray(0, self.face_verts)
        program.setAttributeArray(1, self.face_uvs)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE);
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE);
        GL.glPushAttrib(GL.GL_DEPTH_WRITEMASK)
        GL.glDepthMask(False)
        for i in range(6):
            self.textures[i].bind()
            program.setUniformValue('matrix', self.proj_matrix * self.face_transforms[i])
            GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)
        GL.glPopAttrib(GL.GL_DEPTH_WRITEMASK)


