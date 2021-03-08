import os
from enum import auto

from PyQt5.QtGui import QVector2D, QVector3D, QMatrix4x4
from PyQt5.QtCore import Qt

from wol.CodeEdit import FileCodeNode
from wol.Constants import UserActions, MappingTypes


# Huge potential for becoming a registry anti-pattern. Make sure things added here make sense.
class PlayerContext:
    def __init__(self):
        self.indent = " "*4
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
            (MappingTypes.MouseButtonClicked, Qt.LeftButton): [UserActions.Activate],
            (MappingTypes.KeyDown, Qt.Key_E): [UserActions.Edit],
            (MappingTypes.KeyDown, Qt.Key_Escape): [UserActions.Unselect, UserActions.Release],
            (MappingTypes.KeyDown, Qt.Key_Q): [UserActions.Grab],
            (MappingTypes.KeyDown, Qt.Key_QuoteLeft): [UserActions.Invoke_Console],
            (MappingTypes.KeyDown, Qt.Key_1): [UserActions.Snap_To_90],
            (MappingTypes.KeyDown, Qt.Key_2): [UserActions.Save],
            (MappingTypes.KeyDown, Qt.Key_R): [UserActions.Save],
            (MappingTypes.KeyDown, Qt.Key_L): [UserActions.Load],
            (MappingTypes.KeyPressed, Qt.Key_W): [UserActions.Move_Forward],
            (MappingTypes.KeyPressed, Qt.Key_S): [UserActions.Move_Back],
            (MappingTypes.KeyPressed, Qt.Key_A): [UserActions.Strafe_Left],
            (MappingTypes.KeyPressed, Qt.Key_D): [UserActions.Strafe_Right],
            (MappingTypes.KeyPressed, Qt.Key_Space): [UserActions.Move_Up],
            (MappingTypes.KeyPressed, Qt.Key_Shift): [UserActions.Move_Down],
            (MappingTypes.KeyDown, Qt.Key_C): [UserActions.Copy],
            (MappingTypes.KeyDown, Qt.Key_X): [UserActions.Cut],
            (MappingTypes.KeyDown, Qt.Key_V): [UserActions.Paste],
            (MappingTypes.KeyDown, Qt.Key_Tab): [UserActions.Change_Cursor_Mode],
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
