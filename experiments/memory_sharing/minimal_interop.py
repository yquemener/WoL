# Minimal CUDA/OpenGL zero-copy interop demo
# Creates OpenGL buffer, maps to CUDA, writes with CuPy, renders with OpenGL

from cuda import cudart
import cupy as cp
import numpy as np
import glfw
from OpenGL.GL import *
import torch

if not glfw.init():
    exit()

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
window = glfw.create_window(512, 512, "Minimal Zero-Copy", None, None)
glfw.make_context_current(window)

VERT = """
#version 430
in vec2 pos;
out vec2 uv;
void main() { gl_Position = vec4(pos, 0, 1); uv = pos * 0.5 + 0.5; }
"""

FRAG = """
#version 430
in vec2 uv;
out vec4 col;
layout(std430, binding=0) readonly buffer Buf { float data[]; };
uniform int w;
void main() { 
    int x = int(uv.x * w);
    int y = int(uv.y * w);
    int i = (y * w  + x);
    col = vec4(data[i], data[i], data[i], data[i]); 
}
"""

shader = glCreateProgram()
vs = glCreateShader(GL_VERTEX_SHADER)
glShaderSource(vs, VERT)
glCompileShader(vs)
fs = glCreateShader(GL_FRAGMENT_SHADER)
glShaderSource(fs, FRAG)
glCompileShader(fs)
glAttachShader(shader, vs)
glAttachShader(shader, fs)
glLinkProgram(shader)
glUseProgram(shader)

quad = np.array([-1,-1, 1,-1, -1,1, 1,1], dtype=np.float32)
vao = glGenVertexArrays(1)
glBindVertexArray(vao)
vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo)
glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
glVertexAttribPointer(0, 2, GL_FLOAT, False, 8, None)
glEnableVertexAttribArray(0)

W = 512
ssbo = glGenBuffers(1)
glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
glBufferData(GL_SHADER_STORAGE_BUFFER, W*W*W*4, None, GL_DYNAMIC_DRAW)

flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
gres = cudart.cudaGraphicsGLRegisterBuffer(ssbo, flags)[1]

cudart.cudaGraphicsMapResources(1, gres, None)
err, ptr, size = cudart.cudaGraphicsResourceGetMappedPointer(gres)
mem = cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(ptr, size, None), 0)
img = cp.ndarray((W, W, W), dtype=cp.float32, memptr=mem)
tensor = torch.as_tensor(img, device='cuda')


x = torch.linspace(0, 4*np.pi, W, device='cuda').view(1, -1)
y = torch.linspace(0, 4*np.pi, W, device='cuda').view(-1, 1)    

while not glfw.window_should_close(window):
    t = glfw.get_time()
    tensor[:,:,:] = (torch.cos(x*0.7)+torch.sin(y*2.1+t))
    cudart.cudaDeviceSynchronize()
    cudart.cudaGraphicsUnmapResources(1, gres, None)

    glClear(GL_COLOR_BUFFER_BIT)
    glUniform1i(glGetUniformLocation(shader, "w"), W)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
    
    glfw.swap_buffers(window)
    glfw.poll_events()

cudart.cudaGraphicsUnregisterResource(gres)
glfw.terminate()

