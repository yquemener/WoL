# CudaMemoryNode: SceneNode subclass for GPU memory visualization
# Features:
# - Associates PyTorch tensors with OpenGL buffers for zero-copy rendering
# - Uses CUDA/OpenGL interop via cudaGraphicsGLRegisterBuffer
# - Renders tensor data as textures using shader storage buffers
# Constraints:
# - OpenGL calls must be made from the main Qt thread
# - associate_tensor is thread-safe via @require_gl_thread decorator

from wol.ShadersLibrary import ShadersLibrary
from wol.SceneNode import SceneNode
from PyQt5.QtGui import QImage, QOpenGLTexture, QVector3D
from OpenGL import GL
from wol import utils
from wol.utils import require_gl_thread
from cuda import cudart
import cupy as cp
import torch

class CudaMemoryNode(SceneNode):
    def __init__(self, filename=None, name="Card", parent=None, init_collider=True):
        SceneNode.__init__(self, name, parent)
        self.filename = filename
        if filename:
            self.texture_image = QImage(filename)
        else:
            self.texture_image = None
        self.texture = None

        self.vertices = utils.generate_square_vertices_fan()
        self.texCoords = utils.generate_square_texcoords_fan()
        self.refresh_vertices()
        self.interpolation = GL.GL_LINEAR
        self.buffer_object = None

    def refresh_vertices(self):
        print("refresh_vertices")
        p0 = QVector3D(self.vertices[0][0], self.vertices[0][1], self.vertices[0][2])
        p1 = QVector3D(self.vertices[1][0], self.vertices[1][1], self.vertices[1][2])
        p2 = QVector3D(self.vertices[2][0], self.vertices[2][1], self.vertices[2][2])

    def initialize_gl(self):
        self.program = ShadersLibrary.create_program('cuda_viewer')

    @require_gl_thread(blocking=False)
    def associate_tensor(self, tensor, type_size = 4):
        self.tensor = tensor
        self.buffer_object = GL.glGenBuffers(1)
        glError = GL.glGetError()
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.buffer_object)
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, tensor.numel()*type_size, None, GL.GL_DYNAMIC_DRAW)
        flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
        gres = cudart.cudaGraphicsGLRegisterBuffer(self.buffer_object, flags)[1]
        cudart.cudaGraphicsMapResources(1, gres, None)
        err, ptr, size = cudart.cudaGraphicsResourceGetMappedPointer(gres)
        mem = cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(ptr, size, None), 0)
        img = cp.ndarray(tensor.shape, dtype=cp.float32, memptr=mem)
        tensor.data = torch.as_tensor(img, device='cuda')        

    def paint(self, program):
        # cudart.cudaDeviceSynchronize()
        self.program.bind()
        self.program.setUniformValue("w", 64)
        self.program.setUniformValue('matrix', self.proj_matrix)
        if self.buffer_object is not None:
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 0, self.buffer_object)
            self.program.setAttributeArray(0, self.vertices)
            self.program.setAttributeArray(1, self.texCoords)
            GL.glEnable(GL.GL_BLEND)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, self.interpolation)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, self.interpolation)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)
