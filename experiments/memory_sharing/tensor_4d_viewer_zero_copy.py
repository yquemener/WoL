"""
4D Tensor Slice Viewer - Zero Copy Version
Features:
- Creates 100x100x100x100 PyTorch tensor with sphere geometry and gradient
- Zero-copy GPU rendering using OpenGL Shader Storage Buffer Objects (SSBO)
- Direct 4D tensor access in fragment shaders for slice extraction
- Real-time slice display with mouse controls
- Minimal CPU-GPU memory transfers

Constraints:
- Requires OpenGL 4.3+ for SSBO support
- Tensor uploaded to GPU once, all slicing done in shaders
- 4D indexing: tensor[x, y, z, t] = data[x + y*100 + z*10000 + t*1000000]
"""

import pygame
import numpy as np
import torch
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import sys
import ctypes

class ZeroCopyTensor4DViewer:
    def __init__(self):
        self.width, self.height = 1200, 800
        self.slice_z = 50
        self.slice_y = 50  
        self.slice_t = 50
        
        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("4D Tensor Zero-Copy Viewer")
        
        # Check OpenGL version
        print(f"OpenGL Version: {glGetString(GL_VERSION).decode()}")
        
        self.create_tensor()
        self.setup_opengl()
        self.create_gpu_resources()
        
    def create_tensor(self):
        """Create 4D tensor with sphere and gradient"""
        print("Creating 4D tensor...")
        self.tensor = torch.zeros(100, 100, 100, 100, dtype=torch.float32)
        
        x, y, z = torch.meshgrid(torch.arange(100), torch.arange(100), torch.arange(100), indexing='ij')
        center = 50
        radius = 30
        sphere_mask = ((x - center)**2 + (y - center)**2 + (z - center)**2) <= radius**2
        
        for t in range(100):
            gradient_value = t / 99.0
            self.tensor[:, :, :, t] = sphere_mask.float() * gradient_value
            
        print(f"Tensor shape: {self.tensor.shape}, memory: {self.tensor.numel() * 4 / 1024 / 1024:.1f} MB")
        
    def setup_opengl(self):
        """Setup OpenGL state"""
        glEnable(GL_TEXTURE_2D)
        glViewport(0, 0, self.width, self.height)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        
    def create_gpu_resources(self):
        """Upload tensor to GPU and create shaders"""
        # Upload 4D tensor to SSBO
        print("Uploading tensor to GPU SSBO...")
        tensor_data = self.tensor.numpy().flatten().astype(np.float32)
        
        self.tensor_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.tensor_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, tensor_data.nbytes, tensor_data, GL_STATIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.tensor_ssbo)
        
        # Create shaders for each slice type
        self.create_slice_shaders()
        
        # Create quad for rendering
        self.create_render_quad()
        
    def create_slice_shaders(self):
        """Create shaders for different slice types"""
        
        vertex_shader_source = """
        #version 430 core
        layout (location = 0) in vec2 position;
        layout (location = 1) in vec2 texCoord;
        out vec2 TexCoord;
        uniform vec2 offset;
        uniform vec2 scale;
        void main() {
            gl_Position = vec4((position * scale + offset), 0.0, 1.0);
            TexCoord = texCoord;
        }
        """
        
        # XY slice shader
        xy_fragment_shader = """
        #version 430 core
        in vec2 TexCoord;
        out vec4 FragColor;
        
        layout(std430, binding = 0) buffer TensorData {
            float data[];
        };
        
        uniform int slice_z;
        uniform int slice_t;
        
        int get4DIndex(int x, int y, int z, int t) {
            return x*1000000 + y * 10000 + z * 100 + t;
        }
        
        void main() {
            int x = int(TexCoord.x * 99.0);
            int y = int(TexCoord.y * 99.0);
            int index = get4DIndex(x, y, slice_z, slice_t);
            float value = data[index];
            FragColor = vec4(value, value, value, 1.0);
        }
        """
        
        # XT slice shader  
        xt_fragment_shader = """
        #version 430 core
        in vec2 TexCoord;
        out vec4 FragColor;
        
        layout(std430, binding = 0) buffer TensorData {
            float data[];
        };
        
        uniform int slice_y;
        uniform int slice_z;
        
        int get4DIndex(int x, int y, int z, int t) {
            return x*1000000 + y * 10000 + z * 100 + t;
        }
        
        void main() {
            int x = int(TexCoord.x * 99.0);
            int t = int(TexCoord.y * 99.0);
            int index = get4DIndex(x, slice_y, slice_z, t);
            float value = data[index];
            FragColor = vec4(value, value, value, 1.0);
        }
        """
        
        # XZ slice shader
        xz_fragment_shader = """
        #version 430 core
        in vec2 TexCoord;
        out vec4 FragColor;
        
        layout(std430, binding = 0) buffer TensorData {
            float data[];
        };
        
        uniform int slice_y;
        uniform int slice_t;
        
        int get4DIndex(int x, int y, int z, int t) {
            return x*1000000 + y * 10000 + z * 100 + t;
        }
        
        void main() {
            int x = int(TexCoord.x * 99.0);
            int z = int(TexCoord.y * 99.0);
            int index = get4DIndex(x, slice_y, z, slice_t);
            float value = data[index];
            FragColor = vec4(value, value, value, 1.0);
        }
        """
        
        # Compile shaders
        vertex_shader = compileShader(vertex_shader_source, GL_VERTEX_SHADER)
        
        self.xy_program = compileProgram(vertex_shader, compileShader(xy_fragment_shader, GL_FRAGMENT_SHADER))
        self.xt_program = compileProgram(vertex_shader, compileShader(xt_fragment_shader, GL_FRAGMENT_SHADER))
        self.xz_program = compileProgram(vertex_shader, compileShader(xz_fragment_shader, GL_FRAGMENT_SHADER))
        
        print("Shaders compiled successfully")
        
    def create_render_quad(self):
        """Create quad for rendering slices"""
        vertices = np.array([
            # positions   # texture coords
            -1.0, -1.0,   0.0, 0.0,
             1.0, -1.0,   1.0, 0.0, 
             1.0,  1.0,   1.0, 1.0,
            -1.0,  1.0,   0.0, 1.0
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        self.vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        ebo = glGenBuffers(1)
        
        glBindVertexArray(self.vao)
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))
        glEnableVertexAttribArray(1)
        
    def draw_slice(self, program, viewport, slice_params):
        """Render a slice using specified shader program"""
        glUseProgram(program)
        
        # Set viewport
        glViewport(viewport[0], viewport[1], viewport[2], viewport[3])
        
        # Convert viewport to OpenGL coordinates
        x = (viewport[0] / self.width) * 2 - 1
        y = 1 - ((viewport[1] + viewport[3]) / self.height) * 2
        w = (viewport[2] / self.width) * 2 
        h = (viewport[3] / self.height) * 2
        
        glUniform2f(glGetUniformLocation(program, "offset"), x + w/2, y + h/2)
        glUniform2f(glGetUniformLocation(program, "scale"), w/2, h/2)
        
        # Set slice parameters
        for param, value in slice_params.items():
            glUniform1i(glGetUniformLocation(program, param), value)
            
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
    def handle_mouse(self):
        """Update slice indices based on mouse position"""
        mouse_x, mouse_y = pygame.mouse.get_pos()
        self.slice_z = max(0, min(99, int((mouse_x / self.width) * 99)))
        self.slice_y = max(0, min(99, int((mouse_y / self.height) * 99)))
        self.slice_t = max(0, min(99, int(((mouse_x + mouse_y) / (self.width + self.height)) * 99)))
        
    def run(self):
        """Main rendering loop"""
        clock = pygame.time.Clock()
        running = True
        
        print("Starting render loop...")
        
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                        
            self.handle_mouse()
            
            glClear(GL_COLOR_BUFFER_BIT)
            
            # Calculate slice dimensions
            slice_width = self.width // 3 - 20
            slice_height = min(slice_width, self.height - 150)
            
            # Draw XY slice
            self.draw_slice(
                self.xy_program,
                (10, 100, slice_width, slice_height),
                {"slice_z": self.slice_z, "slice_t": self.slice_t}
            )
            
            # Draw XT slice
            self.draw_slice(
                self.xt_program, 
                (slice_width + 30, 100, slice_width, slice_height),
                {"slice_y": self.slice_y, "slice_z": self.slice_z}
            )
            
            # Draw XZ slice
            self.draw_slice(
                self.xz_program,
                (2 * slice_width + 50, 100, slice_width, slice_height), 
                {"slice_y": self.slice_y, "slice_t": self.slice_t}
            )
            
            # Reset viewport for text rendering
            glViewport(0, 0, self.width, self.height)
            
            pygame.display.flip()
            clock.tick(60)
            print(f"Coordinates: {self.slice_z}, {self.slice_y}, {self.slice_t}")
            
        self.cleanup()
        pygame.quit()
        sys.exit()
        
    def cleanup(self):
        """Clean up GPU resources"""
        glDeleteBuffers(1, [self.tensor_ssbo])
        glDeleteVertexArrays(1, [self.vao])

if __name__ == "__main__":
    viewer = ZeroCopyTensor4DViewer()
    viewer.run()