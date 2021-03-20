from PyQt5.QtGui import QVector3D
from wol.GeomNodes import Sphere
import time


class TestNode(Sphere):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.size=0.15

    def update(self, dt):
        self.position += QVector3D(0, 0, 0)
        print(time.time())
        return

    def new_func(self):
        print("That's the func, baby!")

    def a(t):
        pass

     

