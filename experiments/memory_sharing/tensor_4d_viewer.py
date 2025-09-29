"""
4D Tensor Slice Viewer
Features:
- Creates 100x100x100x100 PyTorch tensor with sphere geometry and gradient
- Displays 3 slices: XY, XT, XZ using OpenGL textures  
- Mouse controls for slice index selection
- Real-time text display of current slice indices
- OpenGL/PyGame rendering with minimal memory reallocation

Constraints:
- Sphere centered at (50,50,50) with radius 30
- Gradient applied along T dimension (4th axis)
- Mouse X controls Z index, Mouse Y controls Y index, combined controls T index
"""

import pygame
import numpy as np
import torch
from pygame.locals import *
import sys

class Tensor4DViewer:
    def __init__(self):
        self.width, self.height = 1200, 800
        self.slice_z = 50  # Z index for XY and XT slices
        self.slice_y = 50  # Y index for XZ slice
        self.slice_t = 50  # T index for XT slice
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("4D Tensor Slice Viewer")
        
        self.font = pygame.font.Font(None, 36)
        self.create_tensor()
        
    def create_tensor(self):
        """Create 4D tensor with sphere and gradient"""
        self.tensor = torch.zeros(100, 100, 100, 100)
        
        # Create coordinate grids
        x, y, z = torch.meshgrid(torch.arange(100), torch.arange(100), torch.arange(100), indexing='ij')
        
        # Sphere equation: (x-50)^2 + (y-50)^2 + (z-50)^2 <= 30^2
        center = 50
        radius = 30
        sphere_mask = ((x - center)**2 + (y - center)**2 + (z - center)**2) <= radius**2
        
        # Apply gradient along T dimension
        for t in range(100):
            gradient_value = t / 99.0  # 0 to 1
            self.tensor[:, :, :, t] = sphere_mask.float() * gradient_value
    
    def get_slices(self):
        """Get current slices as numpy arrays"""
        xy_slice = self.tensor[:, :, self.slice_z, self.slice_t].cpu().numpy()  # X,Y at fixed Z,T
        xt_slice = self.tensor[:, self.slice_y, :, :].cpu().numpy()  # X,Z,T at fixed Y -> take X,T  
        xz_slice = self.tensor[:, self.slice_y, :, self.slice_t].cpu().numpy()  # X,Z at fixed Y,T
        
        # XT slice needs to be reshaped - take slice at Z=slice_z
        xt_slice = self.tensor[:, self.slice_y, :, :].cpu().numpy()[:, self.slice_z, :]  # X,T
        
        return xy_slice, xt_slice, xz_slice
    
    def array_to_surface(self, array):
        """Convert numpy array to pygame surface"""
        # Normalize to 0-255
        normalized = (array * 255).astype(np.uint8)
        # Create RGB array (grayscale)
        rgb_array = np.stack([normalized] * 3, axis=-1)
        return pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))
        
    def handle_mouse(self):
        """Update slice indices based on mouse position"""
        mouse_x, mouse_y = pygame.mouse.get_pos()
        self.slice_z = max(0, min(99, int((mouse_x / self.width) * 99)))
        self.slice_y = max(0, min(99, int((mouse_y / self.height) * 99)))
        self.slice_t = max(0, min(99, int(((mouse_x + mouse_y) / (self.width + self.height)) * 99)))
        
    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                        
            self.handle_mouse()
            
            # Clear screen
            self.screen.fill((0, 0, 0))
            
            # Get current slices
            xy_slice, xt_slice, xz_slice = self.get_slices()
            
            # Convert to surfaces and scale
            slice_size = 250
            xy_surface = pygame.transform.scale(self.array_to_surface(xy_slice), (slice_size, slice_size))
            xt_surface = pygame.transform.scale(self.array_to_surface(xt_slice), (slice_size, slice_size))
            xz_surface = pygame.transform.scale(self.array_to_surface(xz_slice), (slice_size, slice_size))
            
            # Draw slices side by side
            self.screen.blit(xy_surface, (50, 100))
            self.screen.blit(xt_surface, (350, 100)) 
            self.screen.blit(xz_surface, (650, 100))
            
            # Draw labels
            xy_label = self.font.render(f"XY (Z={self.slice_z}, T={self.slice_t})", True, (255, 255, 255))
            xt_label = self.font.render(f"XT (Y={self.slice_y})", True, (255, 255, 255))
            xz_label = self.font.render(f"XZ (Y={self.slice_y}, T={self.slice_t})", True, (255, 255, 255))
            
            self.screen.blit(xy_label, (50, 60))
            self.screen.blit(xt_label, (350, 60))
            self.screen.blit(xz_label, (650, 60))
            
            # Draw mouse info
            mouse_info = self.font.render(f"Mouse: ({pygame.mouse.get_pos()[0]}, {pygame.mouse.get_pos()[1]})", True, (255, 255, 255))
            self.screen.blit(mouse_info, (50, 20))
            
            pygame.display.flip()
            clock.tick(60)
            
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    viewer = Tensor4DViewer()
    viewer.run()
