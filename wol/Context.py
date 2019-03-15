from PyQt5.QtGui import QVector2D, QVector3D


class Context:
    def __init__(self):
        self.abstract_input = dict()
        self.mouse_position = QVector2D(0, 0)
        self.old_mouse_position = QVector2D()
        self.debug_point = QVector3D()
        self.current_camera = None
        self.hover_target = None
        self.focused = None
