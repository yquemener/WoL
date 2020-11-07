# Behaviors are things that are attached to objects and are called on each update. They can hold
# states and describe pieces of behaviors, hence their name
from PyQt5.QtGui import QVector3D, QQuaternion, QMatrix4x4, QMatrix3x3


class Behavior:
    def __init__(self):
        self.obj = None
        self.kill_me = False

    def on_update(self, dt):
        return

    def on_click(self, evt, pos):
        return

    def kill(self):
        self.kill_me = True


# A few built-in behaviors:

# Makes a slerp animation between two positions and orientations
class SlerpAnim(Behavior):
    def __init__(self, delay=1.0,
                 pos1=QVector3D(0, 0, 0), orient1=QQuaternion.fromAxisAndAngle(0, 1, 0, 0),
                 pos2=QVector3D(0, 1, 0), orient2=QQuaternion.fromAxisAndAngle(0, 1, 0, 0)):
        super().__init__()
        self.delay = delay
        self.pos1 = pos1
        self.orient1 = orient1
        self.pos2 = pos2
        self.orient2 = orient2
        self.anim_timer = 0.0

    def on_update(self, dt):
        if self.kill_me:
            return
        self.anim_timer += dt
        alpha = abs(self.anim_timer / self.delay)
        if alpha > 1.0:
            alpha = 1.0
            self.kill_me = True
        self.obj.position = self.pos2*alpha + self.pos1*(1.0-alpha)
        self.obj.orientation = QQuaternion.slerp(self.orient1, self.orient2, alpha)


# Snaps a target to the camera. This behavior assumes it is attached to the camera
class SnapToCamera(Behavior):
    def __init__(self):
        super().__init__()
        self.target = None
        self.target_original_parent = None
        self.target_original_position = None
        self.target_original_orientation = None
        self.anim = None
        self.grabbed_something = False
        self.grab_animation_length = 0.15

    def grab(self, target):
        self.target = target

        # Allows a grab to interrupt an animation and still gets the correct original pose
        b = self.target.get_behavior("SlerpAnim")
        if b is not None:
            self.target.position = b.pos2
            self.target.orientation = b.orient2
            b.kill()

        self.target_original_parent = self.target.parent
        self.target_original_position = self.target.position
        self.target_original_orientation = self.target.orientation
        self.target.compute_transform()
        self.target.reparent(self.obj)
        self.anim = SlerpAnim(self.grab_animation_length,
                              self.target.position, self.target.orientation,
                              QVector3D(0, 0, 5), QQuaternion.fromAxisAndAngle(0, 1, 0, 180))

        self.target.add_behavior(self.anim)
        self.grabbed_something = True

    def restore(self):
        self.target.reparent(self.target_original_parent)
        # Allows a grab to interrupt an animation and still gets the correct original pose
        b = self.target.get_behavior("SlerpAnim")
        if b is not None:
            b.kill()

        self.anim = SlerpAnim(self.grab_animation_length,
                              self.target.position, self.target.orientation,
                              self.target_original_position, self.target_original_orientation)
        self.target.add_behavior(self.anim)
        self.grabbed_something = False


class RotateConstantSpeed(Behavior):
    def __init__(self, speed=10.):
        super().__init__()
        self.speed = speed

    def on_update(self, dt):
        self.obj.orientation *= QQuaternion.fromAxisAndAngle(0, 1, 0, dt*self.speed)


class TransmitClickToParent(Behavior):
    def on_click(self, evt, pos):
        if self.obj.parent is not None:
            self.obj.parent.on_click(evt, pos)
            for b in self.obj.parent.behaviors:
                b.on_click(evt,pos)


class TransmitClickTo(Behavior):
    def __init__(self, target):
        super().__init__()
        self.target = target

    def on_click(self, evt, pos):
        self.target.on_click(evt, pos)
        for b in self.target.behaviors:
            b.on_click(evt,pos)
