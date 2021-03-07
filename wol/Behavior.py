# Behaviors are things that are attached to objects and are called on each update. They can hold
# states and describe pieces of behaviors, hence their name
from collections import defaultdict

from PyQt5.QtGui import QVector3D, QQuaternion, QMatrix4x4, QMatrix3x3

from wol.Constants import Events, UserActions


class Behavior:
    def __init__(self):
        self.obj = None
        self.kill_me = False
        self.events_handlers = defaultdict(list)
        self.init_handlers()

    def init_handlers(self):
        return

    def on_update(self, dt):
        return

    def on_click(self, evt, pos):
        return

    def on_event(self, action):
        print(self, action)
        for h in self.events_handlers[action]:
            h()
        return

    def kill(self):
        self.kill_me = True


# A few built-in behaviors:

# Move the Camera
class MoveAround(Behavior):
    def __init__(self, speed=0.6):
        super().__init__()
        self.speed = speed
        self.events_handlers[UserActions.Move_Up].append(self.on_go_up)
        self.events_handlers[UserActions.Move_Down].append(self.on_go_down)
        self.events_handlers[UserActions.Strafe_Left].append(self.on_go_left)
        self.events_handlers[UserActions.Strafe_Right].append(self.on_go_right)
        self.events_handlers[UserActions.Move_Forward].append(self.on_go_forward)
        self.events_handlers[UserActions.Move_Back].append(self.on_go_back)
        self.reset_movement()

    def reset_movement(self):
        self.go_up = 0
        self.go_down = 0
        self.go_right = 0
        self.go_left = 0
        self.go_forward = 0
        self.go_back = 0

    def on_go_up(self):
        self.go_up = self.speed

    def on_go_down(self):
        self.go_down = self.speed

    def on_go_forward(self):
        self.go_forward = self.speed

    def on_go_back(self):
        self.go_back = self.speed

    def on_go_left(self):
        self.go_left = self.speed

    def on_go_right(self):
        self.go_right = self.speed


    def on_update(self, dt):
        yaw = self.obj.context.mouse_position.x() * 180.0
        pitch = -self.obj.context.mouse_position.y() * 90.0

        xaxis = QVector3D(1, 0, 0)
        yaw_rotation = QQuaternion.fromAxisAndAngle(0, 1, 0, yaw)
        pitch_axis = yaw_rotation.rotatedVector(xaxis)
        pitch_rotation = QQuaternion.fromAxisAndAngle(pitch_axis, pitch)

        m = QMatrix4x4()
        m.rotate(pitch_rotation)
        m.rotate(yaw_rotation)
        direction = m * QVector3D(0, 0, 1)
        self.obj.look_at = self.obj.position + direction
        self.obj.up = QVector3D(0, 1, 0)
        self.obj.orientation = pitch_rotation * yaw_rotation
        right = QVector3D.crossProduct(direction, self.obj.up).normalized()

        delta = QVector3D(0, 0, 0)
        delta += self.obj.up * (self.go_up - self.go_down)
        delta += right * (self.go_right - self.go_left)
        delta += direction * (self.go_forward - self.go_back)
        self.obj.position += delta
        self.obj.look_at += delta
        self.reset_movement()


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
        self.obj.position = self.pos2*alpha + self.pos1*(1.0-alpha)
        self.obj.orientation = QQuaternion.slerp(self.orient1, self.orient2, alpha)
        if alpha == 1.0:
            self.kill_me = True
            self.obj.on_event(Events.AnimationFinished)


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
        self.events_handlers[UserActions.Release].append(self.on_release)
        self.events_handlers[UserActions.Snap_To_90].append(self.snap_to_90)
        self.events_handlers[UserActions.Grab].append(self.on_grab)

    def on_release(self):
        if self.obj.context.focused is not None:
            self.obj.context.focused.focused = False
            self.obj.context.focused.on_event(Events.LostFocus)
            self.obj.context.focused = None
            if self.grabbed_something:
                self.restore()

    def on_grab(self):
        context = self.obj.context
        gr = context.grabbed
        if gr is None:
            if context.hover_target:
                anchor = context.hover_target
                while anchor.properties.get("delegateGrabToParent", False) \
                        and anchor.parent is not None \
                        and anchor.parent is not context.scene:
                    anchor = anchor.parent
                context.grabbed = anchor
                context.grabbed_former_parent = context.grabbed.parent
                context.grabbed.reparent(context.current_camera)
        else:
            context.grabbed.reparent(context.grabbed_former_parent)
            context.grabbed = None

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

    def snap_to_90(self):
        if self.obj.context.grabbed is None:
            return
        g = self.obj.context.grabbed
        euler = g.world_orientation().toEulerAngles()
        euler.setX(round(euler.x() / 90) * 90)
        euler.setY(round(euler.y() / 90) * 90)
        euler.setZ(round(euler.z() / 90) * 90)
        g.set_world_orientation(QQuaternion.fromEulerAngles(euler))


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
