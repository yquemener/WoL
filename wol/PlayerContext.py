import os
from enum import auto

from PyQt5.QtGui import QVector2D, QVector3D, QMatrix4x4
from PyQt5.QtCore import Qt

from wol.CodeEdit import FileCodeNode
from wol.Constants import UserActions, MappingTypes, Events

from wol.SceneNode import instanciate_from_project_file
from wol.SceneNodeEditor import SceneNodeEditor

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
            "rm": self.del_objects,
            "edit": self.edit_object,
            "scene": self.scene,
        }
        self.mappings = {
            (MappingTypes.MouseButtonClicked, Qt.LeftButton): [UserActions.Activate],
            (MappingTypes.KeyDown, Qt.Key_E): [UserActions.Edit],
            (MappingTypes.KeyDown, Qt.Key_Escape): [UserActions.Unselect, UserActions.Release],
            (MappingTypes.KeyDown, Qt.Key_Return): [UserActions.Edit],
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

    def focus(self, target):
        if target == self.focused:
            return
        if self.focused is not None:
            self.focused.on_event(Events.LostFocus)
            self.focused.focused = False

        self.focused = target
        if self.focused is not None:
            self.focused.on_event(Events.GotFocus)
            self.focused.focused = True

    def add_object(self, classname, name=None):
        path = f"my_project/{classname}.py"
        if name is None:
            name = classname
        if os.path.exists(path):
            instanciate_from_project_file(path, classname, {'parent': self.scene})
        else:
            s = f"""
from PyQt5.QtGui import QVector3D
from wol.GeomNodes import Sphere

class {classname}(Sphere):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.size=0.15

    def update(self, dt):
        self.position += QVector3D(0, 0, 0)
"""
            f = open(path, "w")
            f.write(s)
            f.close()
            instanciate_from_project_file(path, classname, {'parent': self.scene, 'name': name})
        return

    def list_objects(self):
        for i, obj in enumerate(self.scene.children):
            if obj.name is not None:
                print(f"{i}: {obj.name}")
            else:
                print(f"{i}: {obj.__class__}@{id(obj)}")
        return

    def del_objects(self, num):
        self.scene.children[num].remove()
        return

    def edit_object(self, num):
        try:
            obj = self.scene.children[num]
            editor = SceneNodeEditor(parent=obj, target=obj, name=f"Editor of {obj.name}")
        except Exception as e:
            print("Failed to edit object:", e)