from collections import defaultdict

from PyQt5.QtCore import Qt, QTimer, QPoint, QSize
from PyQt5.QtGui import QColor, QSurfaceFormat, QVector2D, QVector3D, QCursor, QVector4D, QMatrix3x3, QQuaternion, \
    QMatrix4x4
from PyQt5.QtWidgets import QOpenGLWidget, QApplication
from OpenGL import GL
from math import sin, atan2, tan

from wol.Constants import Events, UserActions
from wol.GuiElements import TextLabelNode
from wol.PlayerContext import PlayerContext, MappingTypes
from wol.ShadersLibrary import ShadersLibrary
from wol.SceneNode import RootNode, SceneNode
from wol.utils import DotDict
from wol.GeomNodes import Sphere