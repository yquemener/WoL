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
from wol.GeomNodes import OdeBoxBehavior
from wol.Constants import Events

class CudaMemoryNode(SceneNode):
    def __init__(self, name="CudaMemoryNode", parent=None, init_collider=True, tensor=None):
        SceneNode.__init__(self, name, parent)

        self.vertices = utils.generate_cube_vertices()
        # self.texCoords = utils.generate_square_texcoords_fan()
        self.texCoords = [[(v[0]+1)/2.0,(v[1]+1)/2.0, (v[2]+1)/2.0] for v in self.vertices]
        self.interpolation = GL.GL_LINEAR
        self.buffer_object = None
        self.value_amplifier = 1.0
        self.properties["skip serialization"] = True
        if tensor is not None:
            self.associate_tensor(tensor)
        self.reshape = None
        self.ode = OdeBoxBehavior(obj=self, kinematic=True)
        self.add_behavior(self.ode)
        self.last_reshape = None

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
        if not self.reshape:
            self.reshape = tensor.shape
            self.on_event(Events.GeometryChanged)

        
    def update(self, dt):
        if self.last_reshape != self.reshape:
            self.last_reshape = self.reshape
            if len(self.reshape) == 1:
                self.scale = QVector3D(self.reshape[0]*0.05, 0.05, 0.05)
            if len(self.reshape) == 2:
                self.scale = QVector3D(self.reshape[0]*0.05, self.reshape[1]*0.05, 0.05)
            if len(self.reshape) == 3:
                self.scale = QVector3D(self.reshape[0]*0.05, self.reshape[1]*0.05, self.reshape[2]*0.05)
            self.on_event(Events.GeometryChanged)

    def paint(self, program):
        # cudart.cudaDeviceSynchronize()
        if self.buffer_object is not None:
            self.program.bind()
            # self.program.setUniformValue("w", self.tensor.shape[-1])
            self.program.setUniformValue("w1", self.reshape[0])
            if len(self.reshape) > 1:
                self.program.setUniformValue("w2", self.reshape[1])    
            else:
                self.program.setUniformValue("w2", 1)
            if len(self.reshape) > 2:
                self.program.setUniformValue("w3", self.reshape[2])
            else:
                self.program.setUniformValue("w3", 1)
            self.program.setUniformValue("value_scale", self.value_amplifier)
            self.program.setUniformValue('matrix', self.proj_matrix)

            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 0, self.buffer_object)
            self.program.setAttributeArray(0, self.vertices)
            self.program.setAttributeArray(1, self.texCoords)
            GL.glEnable(GL.GL_BLEND)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, self.interpolation)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, self.interpolation)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 36)
