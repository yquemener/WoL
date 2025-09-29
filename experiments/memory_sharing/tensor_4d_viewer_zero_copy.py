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
        
        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("4D Tensor Zero-Copy Viewer")
        
        # Check OpenGL version
        print(f"OpenGL Version: {glGetString(GL_VERSION).decode()}")
        
        self.create_tensor()
        self.setup_opengl()
        self.create_gpu_resources()
        
    def create_tensor(self):
        """Create 4D tensor with sphere and gradient using shared buffer (zero-copy)"""
        print("Creating shared 4D tensor buffer...")
        
        # Define tensor dimensions (now fully parametrizable!)
        self.tensor_dims = (150, 100, 100, 100)
        
        # Initialize slice indices based on tensor dimensions
        self.slice_z = self.tensor_dims[2] // 2
        self.slice_y = self.tensor_dims[1] // 2  
        self.slice_t = self.tensor_dims[3] // 2
        
        # Create shared buffer first (numpy array)
        self.shared_buffer = np.zeros(self.tensor_dims, dtype=np.float32)
        
        # Create PyTorch tensor from shared buffer (no copy)
        self.tensor = torch.from_numpy(self.shared_buffer)
        
        print(f"Shared buffer: {self.shared_buffer.shape}, memory: {self.shared_buffer.nbytes / 1024 / 1024:.1f} MB (zero-copy)")
        
        # Fill tensor with sphere and gradient data using actual dimensions
        x, y, z = torch.meshgrid(torch.arange(self.tensor_dims[0]), torch.arange(self.tensor_dims[1]), torch.arange(self.tensor_dims[2]), indexing='ij')
        center_x, center_y, center_z = self.tensor_dims[0]//2, self.tensor_dims[1]//2, self.tensor_dims[2]//2
        radius = min(self.tensor_dims[:3]) // 3
        sphere_mask = ((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2) <= radius**2
        
        for t in range(self.tensor_dims[3]):
            gradient_value = t / (self.tensor_dims[3] - 1.0)
            self.tensor[:, :, :, t] = sphere_mask.float() * gradient_value
        
    def setup_opengl(self):
        """Setup OpenGL state"""
        glEnable(GL_TEXTURE_2D)
        glViewport(0, 0, self.width, self.height)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        
    def create_gpu_resources(self):
        """Upload tensor to GPU and create shaders"""
        # Upload shared buffer to GPU SSBO (zero-copy)
        print("Uploading shared buffer to GPU SSBO...")
        
        # Use ravel() to create flattened view without copying (guaranteed for C-contiguous arrays)
        flattened_buffer = self.shared_buffer.ravel()
        
        self.tensor_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.tensor_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, flattened_buffer.nbytes, flattened_buffer, GL_STATIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.tensor_ssbo)
        
        print(f"Uploaded {flattened_buffer.nbytes / 1024 / 1024:.1f} MB to GPU SSBO (zero-copy)")
        
        # Create universal shader for all slice types
        self.create_universal_shader()
        
        # Create quad for rendering
        self.create_render_quad()
        
    def create_universal_shader(self):
        """Create universal shader for any 4D tensor slice configuration"""
        
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
        
        # Universal fragment shader for any 4D slice configuration
        universal_fragment_shader = """
        #version 430 core
        in vec2 TexCoord;
        out vec4 FragColor;
        
        layout(std430, binding = 0) buffer TensorData {
            float data[];
        };
        
        // Tensor dimensions [dim0, dim1, dim2, dim3]
        uniform ivec4 tensor_dims;
        
        // Mapping: which tensor dimensions correspond to texture coordinates
        uniform int dim_mapping[2];  // dim_mapping[0] = tensor dim for TexCoord.x, dim_mapping[1] = tensor dim for TexCoord.y
        
        // Fixed values for all 4 dimensions (only used for dimensions not in dim_mapping)
        uniform int fixed_values[4]; // fixed_values[i] = fixed value for tensor dimension i
        
        int get4DIndex(int x, int y, int z, int t) {
            // Calculate index using actual tensor dimensions
            return t + z * tensor_dims[3] + y * tensor_dims[3] * tensor_dims[2] + x * tensor_dims[3] * tensor_dims[2] * tensor_dims[1];
        }
        
        void main() {
            // Convert texture coordinates to integer indices using actual dimensions
            int coord_x = int(TexCoord.x * float(tensor_dims[dim_mapping[0]] - 1));
            int coord_y = int(TexCoord.y * float(tensor_dims[dim_mapping[1]] - 1));
            
            // Initialize tensor coordinates with fixed values
            int tensor_coords[4] = int[4](fixed_values[0], fixed_values[1], fixed_values[2], fixed_values[3]);
            
            // Override with variable coordinates based on mapping
            tensor_coords[dim_mapping[0]] = coord_x;
            tensor_coords[dim_mapping[1]] = coord_y;
            
            // Calculate 4D index and fetch value
            int index = get4DIndex(tensor_coords[0], tensor_coords[1], tensor_coords[2], tensor_coords[3]);
            float value = data[index];
            
            FragColor = vec4(value, value, value, 1.0);
        }
        """
        
        # Compile universal shader
        vertex_shader = compileShader(vertex_shader_source, GL_VERTEX_SHADER)
        fragment_shader = compileShader(universal_fragment_shader, GL_FRAGMENT_SHADER)
        
        self.universal_program = compileProgram(vertex_shader, fragment_shader)
        
        print("Universal 4D slice shader compiled successfully")
        
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
        
    def get_slice_config(self, slice_type):
        """Get slice configuration for different slice types"""
        configs = {
            'XY': {
                'dim_mapping': [0, 1],  # X maps to tensor dim 0, Y maps to tensor dim 1
                'fixed_values': [0, 0, self.slice_z, self.slice_t]  # Fixed Z and T
            },
            'XT': {
                'dim_mapping': [0, 3],  # X maps to tensor dim 0, Y maps to tensor dim 3 (T)
                'fixed_values': [0, self.slice_y, self.slice_z, 0]  # Fixed Y and Z
            },
            'XZ': {
                'dim_mapping': [0, 2],  # X maps to tensor dim 0, Y maps to tensor dim 2 (Z)
                'fixed_values': [0, self.slice_y, 0, self.slice_t]  # Fixed Y and T
            },
            'YZ': {
                'dim_mapping': [1, 2],  # X maps to tensor dim 1 (Y), Y maps to tensor dim 2 (Z)
                'fixed_values': [self.slice_z, 0, 0, self.slice_t]  # Fixed X and T
            },
            'YT': {
                'dim_mapping': [1, 3],  # X maps to tensor dim 1 (Y), Y maps to tensor dim 3 (T)
                'fixed_values': [self.slice_z, 0, self.slice_z, 0]  # Fixed X and Z
            },
            'ZT': {
                'dim_mapping': [2, 3],  # X maps to tensor dim 2 (Z), Y maps to tensor dim 3 (T)
                'fixed_values': [self.slice_z, self.slice_y, 0, 0]  # Fixed X and Y
            }
        }
        return configs[slice_type]
        
    def draw_slice(self, slice_type, viewport):
        """Render a slice using universal shader with specified configuration"""
        glUseProgram(self.universal_program)
        
        # Set viewport
        glViewport(viewport[0], viewport[1], viewport[2], viewport[3])
        
        # Convert viewport to OpenGL coordinates
        x = (viewport[0] / self.width) * 2 - 1
        y = 1 - ((viewport[1] + viewport[3]) / self.height) * 2
        w = (viewport[2] / self.width) * 2 
        h = (viewport[3] / self.height) * 2
        
        glUniform2f(glGetUniformLocation(self.universal_program, "offset"), x + w/2, y + h/2)
        glUniform2f(glGetUniformLocation(self.universal_program, "scale"), w/2, h/2)
        
        # Get slice configuration and set uniforms
        config = self.get_slice_config(slice_type)
        
        # Set tensor dimensions
        tensor_dims_loc = glGetUniformLocation(self.universal_program, "tensor_dims")
        glUniform4i(tensor_dims_loc, self.tensor_dims[0], self.tensor_dims[1], self.tensor_dims[2], self.tensor_dims[3])
        
        # Set dimension mapping
        dim_mapping_loc = glGetUniformLocation(self.universal_program, "dim_mapping")
        glUniform1iv(dim_mapping_loc, 2, np.array(config['dim_mapping'], dtype=np.int32))
        
        # Set fixed values
        fixed_values_loc = glGetUniformLocation(self.universal_program, "fixed_values")
        glUniform1iv(fixed_values_loc, 4, np.array(config['fixed_values'], dtype=np.int32))
            
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
    def handle_mouse(self):
        """Update slice indices based on mouse position"""
        mouse_x, mouse_y = pygame.mouse.get_pos()
        self.slice_z = max(0, min(self.tensor_dims[2] - 1, int((mouse_x / self.width) * (self.tensor_dims[2] - 1))))
        self.slice_y = max(0, min(self.tensor_dims[1] - 1, int((mouse_y / self.height) * (self.tensor_dims[1] - 1))))
        self.slice_t = max(0, min(self.tensor_dims[3] - 1, int(((mouse_x + mouse_y) / (self.width + self.height)) * (self.tensor_dims[3] - 1))))
        
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
            slice_height = slice_width
            
            # Draw XY slice
            self.draw_slice('XY', (10, 100, slice_width, slice_height))
            
            # Draw XT slice
            self.draw_slice('XT', (slice_width + 30, 100, slice_width, slice_height))
            
            # Draw XZ slice
            self.draw_slice('XZ', (2 * slice_width + 50, 100, slice_width, slice_height))
            
            # Reset viewport for text rendering
            glViewport(0, 0, self.width, self.height)
            
            pygame.display.flip()
            clock.tick(60)
            
        self.cleanup()
        pygame.quit()
        sys.exit()
        
    def cleanup(self):
        """Clean up GPU resources"""
        glDeleteBuffers(1, [self.tensor_ssbo])
        glDeleteVertexArrays(1, [self.vao])
        glDeleteProgram(self.universal_program)

if __name__ == "__main__":
    viewer = ZeroCopyTensor4DViewer()
    viewer.run()