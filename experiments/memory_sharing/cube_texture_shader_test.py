#!/usr/bin/env python3
# Volume rendering test program using 3D texture and ray marching
# Loads a PyTorch tensor as a 3D texture and renders it using volume ray marching
# Features: box-ray intersection, front-to-back compositing, voxel DDA traversal, slice filtering

import glfw
from OpenGL.GL import *
import numpy as np
import ctypes
import sys
import torch
import argparse

VERTEX_SHADER = """
#version 430 core
layout(location = 0) in vec3 position;

out vec2 fragCoord;

void main() {
    gl_Position = vec4(position, 1.0);
    fragCoord = (position.xy + 1.0) * 0.5 * vec2(800.0, 600.0);
}
"""

FRAGMENT_SHADER = """
#version 430 core
in vec2 fragCoord;
out vec4 fragColor;

layout(std430, binding=0) readonly buffer Buf { float data[]; };

uniform float rotationAngle;
uniform vec2 iResolution;
uniform int w1;
uniform int w2;
uniform int w3;
uniform float value_scale;

const float _Alpha = 0.01;
const vec3 _SliceMin = vec3(-0.91,-0.91,-0.91);
const vec3 _SliceMax = vec3( 0.91, 0.91, 0.91);

float sample3D(vec3 uvw) {
    int x = int(uvw.x * float(w1-1));
    int y = int(uvw.y * float(w2-1));
    int z = int(uvw.z * float(w3-1));
    x = clamp(x, 0, w1-1);
    y = clamp(y, 0, w2-1);
    z = clamp(z, 0, w3-1);
    int i = z*w2*w1 + y*w1 + x;
    return data[i] * value_scale;
}

bool IntersectBox(vec3 ro, vec3 rd, vec3 boxmin, vec3 boxmax, out float tnear, out float tfar)
{
    vec3 invR = 1.0 / rd;
    vec3 tbot = invR * (boxmin.xyz - ro);
    vec3 ttop = invR * (boxmax.xyz - ro);
    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);
    vec2 t0 = max(tmin.xx, tmin.yz);
    tnear = max(t0.x, t0.y);
    t0 = min(tmax.xx, tmax.yz);
    tfar = min(t0.x, t0.y);
    return (tnear > tfar) ? false : true;
}

mat3 RotationY(float y) 
{
    return mat3(cos(y),0.0,-sin(y),0.0,1.0,0.0,sin(y),0.0,cos(y));
}

void main()
{
    vec2 uv = (2.0 * fragCoord.xy - iResolution.xy) / iResolution.y;
    vec3 ro = vec3(0.0, 0.0, -1.8) * RotationY(rotationAngle);
    vec3 rd = normalize(vec3(uv, 2.0)) * RotationY(rotationAngle);
    vec4 color = vec4(0.0,0.0,0.0,0.0);
    float near = 0.0, far = 0.0;
    bool hit = IntersectBox(ro, rd, vec3(-0.5), vec3(0.5), near, far);
    if (hit)
    {
        // Convert entry point from box space [-0.5,0.5] to voxel space [0,w1]x[0,w2]x[0,w3]
        vec3 rayStart = (ro + near * rd + vec3(0.5)) * vec3(w1, w2, w3);
        
        // Current voxel coordinates (integer)
        ivec3 mapPos = ivec3(floor(rayStart));
        
        // Distance (in ray parameter t) to cross one full voxel in each axis
        vec3 deltaDist = abs(vec3(1.0) / rd);
        // Direction to step in each axis: +1 or -1
        ivec3 rayStep = ivec3(sign(rd));
        // Distance (in ray parameter t) to the next voxel boundary in each axis
        vec3 sideDist = (sign(rd) * (vec3(mapPos) - rayStart) + (sign(rd) * 0.5) + 0.5) * deltaDist;
        
        bvec3 mask; // Which axis was crossed (only one component true per iteration)
        int maxSteps = w1 + w2 + w3;
        float tPrev = 0.0; // Track previous boundary distance to compute segment length
        
        for(int i = 0; i < maxSteps; i++)
        {
            // Find which axis boundary will be hit first (mask = true only for that axis)
            mask = lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy));
            // Distance from ray origin to next boundary
            float tNext = dot(sideDist, vec3(mask));
            // Distance traveled through this voxel = difference between boundaries
            float distance = tNext - tPrev;
            
            // Check if current voxel is inside the volume bounds
            if(all(greaterThanEqual(mapPos, ivec3(0))) && all(lessThan(mapPos, ivec3(w1, w2, w3))))
            {
                // Convert voxel coordinates to normalized [0,1] texture coordinates
                vec3 uvw = (vec3(mapPos) + 0.5) / vec3(float(w1), float(w2), float(w3));
                // Sample the weight value at this voxel
                float weight = sample3D(uvw);
                
                // Color: red=negative, green=positive, alpha=magnitude
                vec4 weight_color = vec4(abs(weight), abs(clamp(weight, 0.0, 1.0)), abs(clamp(weight, 0.0, 1.0)), abs(weight));
                // Accumulate color proportional to distance through voxel
                if (mapPos.z == 14) {
                    color += weight_color * distance*10.0;
                }
                else {
                    color += weight_color * distance*0.1;
                }
                color.a = 1.0;
            }
            
            // Advance to next voxel: increment sideDist and mapPos along the crossed axis
            tPrev = tNext;
            sideDist += vec3(mask) * deltaDist;
            mapPos += ivec3(mask) * rayStep;
        }
    }
    
    // Checkerboard background
    vec2 checker_coord = floor(fragCoord / 20.0);
    float checker = mod(checker_coord.x + checker_coord.y, 2.0);
    vec3 bg_color = mix(vec3(0.1,0.1,0.3), vec3(0.2,0.2,0.5), checker);
    
    // Blend volume with background using alpha
    fragColor = vec4(color.rgb + bg_color * (1.0 - color.a), 1.0);
}
"""

def create_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        print(f"Shader compilation error: {error}")
        sys.exit(1)
    return shader

def create_program(vertex_src, fragment_src):
    vertex_shader = create_shader(vertex_src, GL_VERTEX_SHADER)
    fragment_shader = create_shader(fragment_src, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode()
        print(f"Program linking error: {error}")
        sys.exit(1)
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return program

def create_fullscreen_quad():
    return np.array([
        -1, -1, 0,
         1, -1, 0,
         1,  1, 0,
        -1, -1, 0,
         1,  1, 0,
        -1,  1, 0,
    ], dtype=np.float32)

def load_ssbo(filepath, x_range=None, y_range=None, z_range=None):
    tensor = torch.load(filepath)
    data = tensor.detach().cpu().numpy().astype(np.float32)
    print(f"Original shape: {data.shape}")
    if len(data.shape) == 2 and data.shape == (784, 128):
        data = data.reshape(28, 28, 128)
    elif len(data.shape) == 2:
        total = data.shape[0] * data.shape[1]
        if total == 28 * 28 * 128:
            data = data.reshape(28, 28, 128)
    
    if x_range is None:
        x_range = (0, data.shape[0])
    if y_range is None:
        y_range = (0, data.shape[1])
    if z_range is None:
        z_range = (0, data.shape[2])
    
    data = data[x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]]
    print(f"Sliced shape: {data.shape}")
    
    w, h, d = data.shape
    data_flat = data.flatten()
    ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, data_flat.nbytes, data_flat, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo)
    return ssbo, w, h, d, data.min(), data.max()

def parse_slice(slice_str, max_val):
    if ':' in slice_str:
        parts = slice_str.split(':')
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else max_val
        return start, end
    else:
        val = int(slice_str)
        return val, val + 1

def main():
    parser = argparse.ArgumentParser(description='Volume rendering with slice filtering')
    parser.add_argument('--slice', type=str, default=':', help='Slice range in format x1:x2,y1:y2,z1:z2 (e.g. 1:5,1:5,1:10)')
    args = parser.parse_args()
    
    if not glfw.init():
        sys.exit(1)
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    
    window = glfw.create_window(800, 600, "Volume Rendering Test", None, None)
    if not window:
        glfw.terminate()
        sys.exit(1)
    
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    
    mouse_x = [400.0]
    
    def cursor_callback(window, xpos, ypos):
        mouse_x[0] = xpos
    
    glfw.set_cursor_pos_callback(window, cursor_callback)
    
    program = create_program(VERTEX_SHADER, FRAGMENT_SHADER)
    vertices = create_fullscreen_quad()
    
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    
    glClearColor(0.0, 0.0, 0.0, 1.0)
    
    slice_parts = args.slice.split(',')
    if len(slice_parts) == 1 and slice_parts[0] == ':':
        x_range, y_range, z_range = None, None, None
    elif len(slice_parts) == 3:
        temp_tensor = torch.load("layer1.pt").detach().cpu().numpy()
        if len(temp_tensor.shape) == 2 and temp_tensor.shape == (784, 128):
            temp_tensor = temp_tensor.reshape(28, 28, 128)
        elif len(temp_tensor.shape) == 2:
            total = temp_tensor.shape[0] * temp_tensor.shape[1]
            if total == 28 * 28 * 128:
                temp_tensor = temp_tensor.reshape(28, 28, 128)[:,:,:28]
        w_full, h_full, d_full = temp_tensor.shape
        x_range = parse_slice(slice_parts[0], w_full)
        y_range = parse_slice(slice_parts[1], h_full)
        z_range = parse_slice(slice_parts[2], d_full)
        print(f"Loading slice: X[{x_range[0]}:{x_range[1]}], Y[{y_range[0]}:{y_range[1]}], Z[{z_range[0]}:{z_range[1]}]")
    else:
        print("Invalid slice format. Use x1:x2,y1:y2,z1:z2")
        sys.exit(1)
    
    ssbo, w, h, d, vmin, vmax = load_ssbo("layer1.pt", x_range, y_range, z_range)
    value_scale = 1.0 / (vmax - vmin) if vmax > vmin else 1.0
    
    rotation_location = glGetUniformLocation(program, "rotationAngle")
    iResolution_location = glGetUniformLocation(program, "iResolution")
    w1_location = glGetUniformLocation(program, "w1")
    w2_location = glGetUniformLocation(program, "w2")
    w3_location = glGetUniformLocation(program, "w3")
    value_scale_location = glGetUniformLocation(program, "value_scale")
    
    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)
        
        rotation_angle = (mouse_x[0] / 800.0) * 6.28318
        
        glUseProgram(program)
        glUniform1f(rotation_location, rotation_angle)
        glUniform2f(iResolution_location, 800.0, 600.0)
        glUniform1i(w1_location, w)
        glUniform1i(w2_location, h)
        glUniform1i(w3_location, d)
        glUniform1f(value_scale_location, value_scale)
        
        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        
        glfw.swap_buffers(window)
        glfw.poll_events()
    
    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo, ssbo])
    glDeleteProgram(program)
    glfw.terminate()

if __name__ == "__main__":
    main()

