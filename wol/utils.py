from PyQt5.QtGui import QVector3D, QMatrix4x4
from PyQt5.QtCore import QCoreApplication, QThread
from PyQt5.QtWidgets import QApplication
import threading
import sys
import trace
from functools import wraps
import queue


_gl_call_queue = queue.Queue()

def process_gl_queue():
    while not _gl_call_queue.empty():
        try:
            func, args, kwargs, result_queue = _gl_call_queue.get_nowait()
            try:
                result = func(*args, **kwargs)
                if result_queue is not None:
                    result_queue.put(('success', result))
            except Exception as e:
                if result_queue is not None:
                    result_queue.put(('error', e))
        except queue.Empty:
            break

def require_gl_thread(blocking=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            app = QApplication.instance()
            if app is None:
                return func(*args, **kwargs)
            
            current_thread = QThread.currentThread()
            main_thread = app.thread()
            
            if current_thread == main_thread:
                return func(*args, **kwargs)
            else:
                result_queue = queue.Queue() if blocking else None
                _gl_call_queue.put((func, args, kwargs, result_queue))
                
                if blocking:
                    status, result = result_queue.get()
                    if status == 'error':
                        raise result
                    return result
                return None
        return wrapper
    return decorator

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class KillableThread(threading.Thread):
    def __init__(self, target, args):
        super().__init__(target=target, args=args)
        self.kill_me = False
        # threading.settrace(self.trace_func)

    def trace_func(self, frame, event, arg):
        if self.kill_me:
            raise SystemExit()

    def start(self):
        self.__run_backup = self.run
        self.run = self.__run
        threading.Thread.start(self)

    def __run(self):
        sys.settrace(self.trace_func)
        self.__run_backup()
        self.run = self.__run_backup

    def kill(self):
        self.kill_me = True


def generate_square_vertices_fan():
    vertices = list()
    vertices.append([-1.0, -1.0, 0.0])
    vertices.append([+1.0, -1.0, 0.0])
    vertices.append([+1.0, +1.0, 0.0])
    vertices.append([-1.0, +1.0, 0.0])
    return vertices


def generate_square_texcoords_fan():
    tex_coords = list()
    tex_coords.append([0.0, 1.0])
    tex_coords.append([1.0, 1.0])
    tex_coords.append([1.0, 0.0])
    tex_coords.append([0.0, 0.0])
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


def generate_cube_vertices():
    # v1  v2        v5  v6
    #
    # v3  v4        v7  v8

    v1 = QVector3D(-1, 1, -1)
    v2 = QVector3D(1,  1, -1)
    v3 = QVector3D(-1, 1,  1)
    v4 = QVector3D(1,  1,  1)
    v5 = QVector3D(-1, -1, -1)
    v6 = QVector3D(1, -1,  -1)
    v7 = QVector3D(-1, -1,  1)
    v8 = QVector3D(1, -1,   1)

    vertices = list()
    vertices += [v1, v2, v3, ]
    vertices += [v3, v2, v4, ]
    vertices += [v5, v6, v7, ]
    vertices += [v7, v6, v8, ]
    vertices += [v3, v4, v7, ]
    vertices += [v7, v4, v8, ]
    vertices += [v1, v2, v5, ]
    vertices += [v5, v2, v6, ]
    vertices += [v1, v5, v3, ]
    vertices += [v3, v5, v7, ]
    vertices += [v2, v6, v4, ]
    vertices += [v4, v6, v8, ]
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
