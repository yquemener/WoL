import os
from enum import auto

from PyQt5.QtGui import QVector2D, QVector3D, QMatrix4x4
from PyQt5.QtCore import Qt

from wol.CodeEdit import FileCodeNode


class UserActions(int):
    Edit = 1
    Activate = 2
    Unselect = 3
    Grab = 4
    Release = 5
    Invoke_Console = 6
    Save = 6
    Snap_To_90 = 7
    Move_Forward = 8
    Move_Back = 9
    Strafe_Left = 10
    Strafe_Right = 11


class MappingTypes(int):
    Key = 1
    Mouse_Button = 2


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
        self.project_dir = os.getcwd() + "/my_project/"
        self.execution_context = {
            "add": self.add_object,
            "ls": self.list_objects,
            "rm": self.del_objects
        }
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

    def add_object(self, name=None):
        if name is None:
            nums = [int(fn.split("_")[1].split(".")[0]) for fn in os.listdir(self.project_dir) if fn.startswith("cell_")]
            name = f"{self.project_dir}/cell_{max(nums)+1}.py"
        newobj = FileCodeNode(parent=self.scene, filename=name)
        return

    def list_objects(self):
        for i, obj in enumerate(self.scene.children):
            print(f"{i}: {obj.name}")
        return

    def del_objects(self, num):
        self.scene.children[num].remove()
        return
