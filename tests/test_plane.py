from unittest import TestCase

from PyQt5.QtGui import QVector3D

from wol.Collisions import Ray, Plane


class TestPlane(TestCase):
    def test_collide_with_ray(self):
        ray = Ray(pos=QVector3D(10, 10, 10),
                  direction=QVector3D(0, -1, 0))
        ray2 = Ray(pos=QVector3D(10, 10, 10),
                  direction=QVector3D(1, 0, 0))
        ray3 = Ray(pos=QVector3D(10, 10, 10),
                  direction=QVector3D(0, 1, 0))
        plane = Plane(normale=QVector3D(0, 1, 0))

        self.assertTrue(plane.collide_with_ray(ray)[0])
        self.assertFalse(plane.collide_with_ray(ray2)[0])
        self.assertFalse(plane.collide_with_ray(ray3)[0])
