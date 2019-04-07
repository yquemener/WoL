from PyQt5.QtGui import QVector3D, QMatrix4x4


def generate_square_vertices_fan():
    vertices = list()
    vertices.append([-1.0, -1.0, 0.0])
    vertices.append([+1.0, -1.0, 0.0])
    vertices.append([+1.0, +1.0, 0.0])
    vertices.append([-1.0, +1.0, 0.0])
    return vertices


def generate_square_texcoords_fan():
    tex_coords = list()
    tex_coords.append([1.0, 1.0])
    tex_coords.append([0.0, 1.0])
    tex_coords.append([0.0, 0.0])
    tex_coords.append([1.0, 0.0])
    return tex_coords


def generate_square_vertices_triangles():
    vertices = list()
    vertices.append([-1.0, -1.0, 0.0])
    vertices.append([+1.0, -1.0, 0.0])
    vertices.append([+1.0, +1.0, 0.0])
    vertices.append([-1.0, -1.0, 0.0])
    vertices.append([+1.0, +1.0, 0.0])
    vertices.append([-1.0, +1.0, 0.0])
    return vertices


def generate_square_texcoords_triangles():
    tex_coords = list()
    tex_coords.append([1.0, 1.0])
    tex_coords.append([0.0, 1.0])
    tex_coords.append([0.0, 0.0])
    tex_coords.append([1.0, 1.0])
    tex_coords.append([0.0, 0.0])
    tex_coords.append([1.0, 0.0])
    return tex_coords


def vectorize(list_of_triples):
    vectors = list()
    for v in list_of_triples:
        vectors.append(QVector3D(v[0], v[1], v[2]))
    return vectors


def transformed_buffer(buffer, transform=QMatrix4x4()):
    ret = list()
    for v in buffer:
        ret.append(transform.map(v))
    return ret


def transform_buffer(buffer, transform=QMatrix4x4()):
    for i in range(len(buffer)):
        buffer[i] = transform.map(buffer[i])
