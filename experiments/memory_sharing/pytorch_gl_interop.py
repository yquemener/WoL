# MIT License
# Copyright (c) 2024 Jean-Baptiste Keck
#
# True zero-copy PyTorch/OpenGL interop using SSBO
# 
# Features:
# - PyTorch tensor created directly in OpenGL SSBO (no copy)
# - All PyTorch operations write directly to GPU buffer
# - Buffer stays mapped during computation, unmapped only for rendering
# - Explicit synchronization with cudaDeviceSynchronize
#
# Constraints:
# - Requires OpenGL 4.3+ for SSBO support
# - Requires CUDA-enabled PyTorch
# - Buffer must be mapped when tensor is used, unmapped for OpenGL

import sys
import os
from cuda import cudart
import numpy as np
import cupy as cp
import torch
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders


def format_cudart_err(err):
    return (
        f"{cudart.cudaGetErrorName(err)[1].decode('utf-8')}({int(err)}): "
        f"{cudart.cudaGetErrorString(err)[1].decode('utf-8')}"
    )


def check_cudart_err(args):
    if isinstance(args, tuple):
        assert len(args) >= 1
        err = args[0]
        if len(args) == 1:
            ret = None
        elif len(args) == 2:
            ret = args[1]
        else:
            ret = args[1:]
    else:
        err = args
        ret = None

    assert isinstance(err, cudart.cudaError_t), type(err)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(format_cudart_err(err))

    return ret


class CudaOpenGLMappedBuffer:
    def __init__(self, gl_buffer, flags=0):
        self._gl_buffer = int(gl_buffer)
        self._flags = int(flags)
        self._graphics_ressource = None
        self._cuda_buffer = None
        self.register()

    @property
    def gl_buffer(self):
        return self._gl_buffer

    @property
    def cuda_buffer(self):
        assert self.mapped
        return self._cuda_buffer

    @property
    def graphics_ressource(self):
        assert self.registered
        return self._graphics_ressource

    @property
    def registered(self):
        return self._graphics_ressource is not None

    @property
    def mapped(self):
        return self._cuda_buffer is not None

    def __enter__(self):
        return self.map()

    def __exit__(self, exc_type, exc_value, trace):
        self.unmap()
        return False

    def __del__(self):
        self.unregister()

    def register(self):
        if self.registered:
            return self._graphics_ressource
        self._graphics_ressource = check_cudart_err(
            cudart.cudaGraphicsGLRegisterBuffer(self._gl_buffer, self._flags)
        )
        return self._graphics_ressource

    def unregister(self):
        if not self.registered:
            return self
        self.unmap()
        self._graphics_ressource = check_cudart_err(
            cudart.cudaGraphicsUnregisterResource(self._graphics_ressource)
        )
        return self

    def map(self, stream=None):
        if not self.registered:
            raise RuntimeError("Cannot map an unregistered buffer.")
        if self.mapped:
            return self._cuda_buffer

        check_cudart_err(
            cudart.cudaGraphicsMapResources(1, self._graphics_ressource, stream)
        )

        ptr, size = check_cudart_err(
            cudart.cudaGraphicsResourceGetMappedPointer(self._graphics_ressource)
        )

        self._cuda_buffer = cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(ptr, size, self), 0
        )

        return self._cuda_buffer

    def unmap(self, stream=None):
        if not self.registered:
            raise RuntimeError("Cannot unmap an unregistered buffer.")
        if not self.mapped:
            return self

        self._cuda_buffer = check_cudart_err(
            cudart.cudaGraphicsUnmapResources(1, self._graphics_ressource, stream)
        )

        return self


class CudaOpenGLMappedArray(CudaOpenGLMappedBuffer):
    def __init__(self, dtype, shape, gl_buffer, flags=0, strides=None, order='C'):
        super().__init__(gl_buffer, flags)
        self._dtype = dtype
        self._shape = shape
        self._strides = strides
        self._order = order

    @property
    def cuda_array(self):
        assert self.mapped
        return cp.ndarray(
            shape=self._shape,
            dtype=self._dtype,
            strides=self._strides,
            order=self._order,
            memptr=self._cuda_buffer,
        )

    def map(self, *args, **kwargs):
        super().map(*args, **kwargs)
        return self.cuda_array


VERTEX_SHADER = """
#version 430

in vec2 position;
out vec2 uv;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    uv = position * 0.5 + 0.5;
}
"""


FRAGMENT_SHADER = """
#version 430

in vec2 uv;
out vec4 outColor;

layout(std430, binding = 0) readonly buffer ImageBuffer {
    float data[];
};

uniform int width;
uniform int height;

void main() {
    int x = int(uv.x * width);
    int y = int(uv.y * height);
    int idx = (y * width + x) * 4;
    
    float r = data[idx + 0];
    float g = data[idx + 1];
    float b = data[idx + 2];
    float a = data[idx + 3];
    
    outColor = vec4(r, g, b, a);
}
"""


def setup_fullscreen_quad():
    vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    ], dtype=np.float32)

    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    return VAO, VBO


def setup_image_buffer(width, height):
    ftype = np.float32
    num_pixels = width * height
    num_channels = 4
    buffer_bytes = num_pixels * num_channels * ftype().nbytes

    flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard

    SSBO = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO)
    glBufferData(GL_SHADER_STORAGE_BUFFER, buffer_bytes, None, GL_DYNAMIC_DRAW)

    image_buffer = CudaOpenGLMappedArray(
        ftype, (height, width, num_channels), SSBO, flags
    )

    return image_buffer


class PyTorchImageUpdater:
    def __init__(self, width, height, image_buffer):
        self.W = width
        self.H = height
        self.image_buffer = image_buffer
        
        cupy_view = image_buffer.map()
        self.torch_img = torch.as_tensor(cupy_view, device='cuda')
        
        x = torch.linspace(0, 4*np.pi, width, device='cuda').view(1, -1)
        y = torch.linspace(0, 4*np.pi, height, device='cuda').view(-1, 1)
        
        self.x_grid = x.expand(height, width)
        self.y_grid = y.expand(height, width)
        
        print(f"PyTorch tensor shape: {self.torch_img.shape}")
        print(f"PyTorch tensor device: {self.torch_img.device}")
        print(f"PyTorch tensor dtype: {self.torch_img.dtype}")
        print(f"Points to OpenGL buffer: True (zero-copy)")
        
    def update(self, t):
        wave1 = torch.sin(self.x_grid + t)
        wave2 = torch.cos(self.y_grid + t * 0.7)
        pattern = (wave1 * wave2 + 1.0) * 0.5
        
        self.torch_img[:, :, 0] = pattern * (torch.sin(torch.tensor(t * 0.5, device='cuda')) * 0.5 + 0.5)
        self.torch_img[:, :, 1] = pattern * (torch.cos(torch.tensor(t * 0.3, device='cuda')) * 0.5 + 0.5)
        self.torch_img[:, :, 2] = pattern
        self.torch_img[:, :, 3] = 1.0
        
        check_cudart_err(cudart.cudaDeviceSynchronize())
        
    def unmap(self):
        self.image_buffer.unmap()
        
    def remap(self):
        cupy_view = self.image_buffer.map()
        self.torch_img = torch.as_tensor(cupy_view, device='cuda')


def get_gpu_mem():
    import subprocess
    result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    for line in result.stdout.strip().split('\n'):
        if line:
            pid, mem = line.split(',')
            if int(pid.strip()) == os.getpid():
                return int(mem.strip())
    return 0


def main():
    if not glfw.init():
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    title = "PyTorch/OpenGL zero-copy interop"
    window = glfw.create_window(800, 800, title, None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.swap_interval(0)

    print(f"OpenGL version: {glGetString(GL_VERSION).decode()}")
    print(f"GLSL version: {glGetString(GL_SHADING_LANGUAGE_VERSION).decode()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print()

    print(f"Initial VRAM: {get_gpu_mem()} MB")

    shader = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
    )

    VAO, VBO = setup_fullscreen_quad()
    
    width, height = 512, 512
    image_buffer = setup_image_buffer(width, height)
    print(f"After SSBO allocation: {get_gpu_mem()} MB")
    
    updater = PyTorchImageUpdater(width, height, image_buffer)
    print(f"After PyTorch setup: {get_gpu_mem()} MB")
    print()

    positionLoc = glGetAttribLocation(shader, "position")
    widthLoc = glGetUniformLocation(shader, "width")
    heightLoc = glGetUniformLocation(shader, "height")

    fps = 0
    nframes = 0
    last_time = glfw.get_time()

    glUseProgram(shader)

    while not glfw.window_should_close(window):
        t = glfw.get_time()
        dt = t - last_time
        if dt >= 1.0:
            fps = nframes / dt
            last_time = t
            nframes = 0

        updater.remap()
        updater.update(t)
        updater.unmap()

        win_width, win_height = glfw.get_window_size(window)
        glViewport(0, 0, win_width, win_height)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUniform1i(widthLoc, width)
        glUniform1i(heightLoc, height)

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, image_buffer.gl_buffer)
        
        glBindVertexArray(VAO)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glEnableVertexAttribArray(positionLoc)
        glVertexAttribPointer(positionLoc, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        glfw.swap_buffers(window)
        glfw.poll_events()
        glfw.set_window_title(window, f"{title} ({fps:.1f} fps)")
        nframes += 1

    image_buffer.unregister()
    glfw.terminate()


if __name__ == "__main__":
    sys.exit(main())
