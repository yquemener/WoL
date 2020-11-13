from PyQt5.QtGui import QVector2D, QVector3D, QMatrix4x4
from PyQt5.QtCore import Qt


class UserActions(int):
    Edit = ...
    Activate = ...
    Unselect = ...
    Grab = ...
    Release = ...
    Invoke_Console = ...
    Save = ...
    Snap_To_90 = ...
    Move_Forward = ...
    Move_Back = ...
    Strafe_Left = ...
    Strafe_Right = ...


class MappingTypes(int):
    Key = ...
    Mouse_Button = ...


# Huge potential for becoming a registry anti-pattern. Make sure things added here make sense.
class PlayerContext:
    def __init__(self):
        self.abstract_input = dict()
        self.mouse_position = QVector2D(0, 0)
        self.old_mouse_position = QVector2D()
        self.debug_point = QVector3D()
        self.current_camera = None
        self.hover_target = None
        self.focused = None
        self.scene = None
        self.grabbed = None
        self.grabbed_transform = QMatrix4x4()
        self.mappings = {
            (MappingTypes.Mouse_Button, Qt.LeftButton): [UserActions.Activate],
            (MappingTypes.Key, Qt.Key_E): [UserActions.Edit],
            (MappingTypes.Key, Qt.Key_Escape): [UserActions.Unselect, UserActions.Release],
            (MappingTypes.Key, Qt.Key_Q): [UserActions.Grab],
            (MappingTypes.Key, Qt.Key_QuoteLeft): [UserActions.Invoke_Console],
            (MappingTypes.Key, Qt.Key_1): [UserActions.Snap_To_90],
            (MappingTypes.Key, Qt.Key_2): [UserActions.Save],
            (MappingTypes.Key, Qt.Key_W): [UserActions.Move_Forward],
            (MappingTypes.Key, Qt.Key_S): [UserActions.Move_Back],
            (MappingTypes.Key, Qt.Key_A): [UserActions.Strafe_Left],
            (MappingTypes.Key, Qt.Key_D): [UserActions.Strafe_Right],
        }

