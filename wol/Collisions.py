from PyQt5.QtGui import QVector3D, QVector2D, QMatrix4x4

epsilon = 1e-6


class Line2D:
    def __init__(self, pos=QVector2D(0, 0), dir=QVector2D(1, 0)):
        self.pos = pos
        self.dir = dir.normalized()


class Ray:
    def __init__(self, pos=QVector3D(0, 0, 0), dir=QVector3D(1, 0, 0)):
        self.pos = pos
        self.dir = dir.normalized()


class Plane:
    def __init__(self, point=QVector3D(), normale=QVector3D(1, 0, 0)):
        self.orig = point
        self.normale = normale.normalized()
        self.u = QVector3D(0, 0, 1)
        self.v = QVector3D(0, 1, 0)
        self.transform = QMatrix4x4()

    def update_coord_system(self):
        x = QVector3D(1, 0, 0)
        if abs(QVector3D.dotProduct(x, self.normale))<0.01:
            ref_u = QVector3D(0, 0, 1)
        else:
            ref_u = QVector3D(1, 0, 0)
        self.v = QVector3D.crossProduct(ref_u, self.normale).normalized()
        self.u = QVector3D.crossProduct(self.normale, self.v).normalized()

    def project_2d(self, point_):
        point = self.transform.inverted()[0].map(point_)
        return QVector2D(
            QVector3D.dotProduct((point - self.orig), self.u),
            QVector3D.dotProduct((point - self.orig), self.v))

    def collide_with_ray(self, ray_, double_sided=True):
        ray = Ray(self.transform.inverted()[0].map(ray_.pos),
                  self.transform.inverted()[0].mapVector(ray_.dir))

        # Looking toward each other?
        if QVector3D.dotProduct(self.normale, ray.dir) > 0 and not double_sided:
            return False, None

        # Distance from ray.pos to the plane
        dist = QVector3D.dotProduct(self.normale, (self.orig - ray.pos))
        if dist > 0 and not double_sided:
            return False, None

        factor = QVector3D.dotProduct(ray.dir, self.normale)
        if abs(factor) < epsilon:
            return False, None
        proj = ray.pos + dist * ray.dir / factor

        # Check it is in the direction of the ray
        if QVector3D.dotProduct(proj-ray.pos, ray.dir)<0:
            return False, None

        return True, self.transform.map(proj)


class Outline3D(Plane):     # Flat polygon
    def __init__(self, point=QVector3D(), normale=QVector3D(1,0,0)):
        Plane.__init__(self, point, normale)
        # list of QVector2D
        self.points = list()

    def add_2d_point(self, point):
        self.points.append(point)

    def add_3d_point(self, point):
        self.points.append(self.project_2d(point))

    def collide_with_ray(self, ray, double_sided=True):
        v, inters = Plane.collide_with_ray(self, ray, double_sided=double_sided)
        if not v:
            return False, None
        inters2d = self.project_2d(inters)
        return is_inside_outline_2d(self.points, inters2d), inters


def collision_line_line_2d(line1, line2):
    #
    normale2 = QVector2D(line2.dir[1], -line2.dir[0])
    if abs(QVector2D.dotProduct(normale2, line1.dir)) < epsilon:
        return False, None

    d = QVector2D.dotProduct(line2.pos - line1.pos, normale2)
    f = QVector2D.dotProduct(line1.dir, normale2)
    proj = line1.pos + line1.dir*d/f
    return True, proj


def collision_segment_segment_2d(seg1, seg2):
    l1 = Line2D(seg1[0], seg1[1] - seg1[0])
    l2 = Line2D(seg2[0], seg2[1] - seg2[0])
    v, inters = collision_line_line_2d(l1,l2)
    if not v:
        return False, None

    # Project intersection on seg1
    p1 = QVector2D.dotProduct(inters-l1.pos, l1.dir)
    if p1 < 0 or p1 > (seg1[1]-seg1[0]).length():
        return False, None

    # Project intersection on seg2
    p2 = QVector2D.dotProduct(inters-l2.pos, l2.dir)
    if p2 < 0 or p2 > (seg2[1]-seg2[0]).length():
        return False, None

    return True, inters


def collision_ray_segment_2d(ray, seg):
    v, inters = collision_line_line_2d(Line2D(ray.pos, ray.dir),
                                       Line2D(seg[0], seg[1]-seg[0]))
    if not v:
        return False, None

    if QVector2D.dotProduct(inters-ray.pos, ray.dir) < 0:
        # The intersection point is behind the ray origin
        return False, None

    # Project intersection on the segment
    p = QVector2D.dotProduct(inters-seg[0], (seg[1]-seg[0]).normalized())
    if p < 0 or p > (seg[1]-seg[0]).length():
        return False, None

    return True, inters


def collisions_ray_outline_2d(ray, outline):
    ret = list()
    for i in range(len(outline)):
        seg = [outline[i], outline[(i+1) % len(outline)]]
        v, inters = collision_ray_segment_2d(ray, seg)
        if v:
            ret.append(inters)
    return ret


def is_inside_outline_2d(outline, point):
    ray = Ray(point, QVector2D(0.5, 1.24).normalized())
    r = collisions_ray_outline_2d(ray, outline)
    if len(r) % 2 == 1:
        return True
    else:
        return False





def collision_ray(ray, obj, transform):
    if isinstance(obj, Outline3D):
        return collision_ray_outline_3d(ray, obj, transform=transform, double_sided=True)
    if isinstance(obj, Plane):
        return collision_ray_plane(ray, obj, transform)
    raise NotImplementedError("Collision between a ray and a "+str(type(obj))+" has not been implemented")
