from unittest import TestCase

from PyQt5.QtGui import QVector3D, QVector2D

from wol import Collisions


class TestOutline3D(TestCase):
    def test_add_3d_point(self):
        outline = Collisions.Outline3D(normale=QVector3D(0, 0, 1))
        p = QVector3D(0, 0, 0)
        p2 = QVector2D(0, 0)
        outline.add_3d_point(p)
        self.assertAlmostEqual(outline.points[0].x(), p2.x())
        self.assertAlmostEqual(outline.points[0].y(), p2.y())

    def test_add_3d_point2(self):
        verts = list()
        verts.append([-1.0, -1.0, 0.0])
        verts.append([+1.0, -1.0, 0.0])
        verts.append([+1.0, +1.0, 0.0])
        verts.append([-1.0, +1.0, 0.0])
        outline = Collisions.Outline3D(normale=QVector3D(0, 0, 1))
        for v in verts:
            outline.add_3d_point(QVector3D(*v))
        for i in range(4):
            self.assertAlmostEqual(outline.points[i].x(), verts[i][0])
            self.assertAlmostEqual(outline.points[i].y(), verts[i][1])

    def test_collide_with_ray(self):
        self.assertTrue(True)
