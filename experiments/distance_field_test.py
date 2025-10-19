#!/usr/bin/env python3
# Standalone ray-marching test program
# Renders a sphere using ray-marching technique in fragment shader
# The cube vertices are used as a bounding box for ray entry points

import glfw
from OpenGL.GL import *
import numpy as np
import ctypes
import sys
from math import sin, cos

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoord;

uniform mat4 mvp;
uniform mat4 model;
out vec3 worldPos;

void main() {
    gl_Position = mvp * vec4(position, 1.0);
    worldPos = (model * vec4(position, 1.0)).xyz;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 worldPos;
out vec4 fragColor;

uniform vec3 cameraPos;


float sdBox( vec3 p, vec3 b )
{
    vec3 d = abs(p) - b;
    return length(max(d,0.0));
}

float sdShape(vec3 p, float radius) {
    // actually a grid
    vec3 b = vec3(0.03,0.03,0.03);
    vec3 p2 = p*10.0;
    float offset = 0.0;
    vec3 pr = vec3(p2.x - round(p2.x), p2.y-round(p2.y), p2.z);
    float res = 0.0;
    float bb = sdBox(p, vec3(0.5,0.5,0.5));
    if ((int(round(p2.x))*int(round(p2.y)))%2 == 1){
        vec3 pa = vec3(pr.x +1.0, pr.yz);
        vec3 pb = vec3(pr.x -1.0, pr.yz);
        vec3 pc = vec3(pr.x, pr.y+1.0, pr.z);
        vec3 pd = vec3(pr.x, pr.y-1.0, pr.z);
        res = min(min(  sdBox(pa/10.0, b),
                        sdBox(pb/10.0, b)),
                   min(    
                    sdBox(pc/10.0, b),
                    sdBox(pd/10.0, b)));
        return max(bb, res);
    }
    
    return max(bb, sdBox(pr/10.0, b));
}

vec3 calcNormal(vec3 p) {
    float eps = 0.001;
    float d = sdShape(p, 1.0);
    return normalize(vec3(
        sdShape(p + vec3(eps, 0, 0), 1.0) - d,
        sdShape(p + vec3(0, eps, 0), 1.0) - d,
        sdShape(p + vec3(0, 0, eps), 1.0) - d
    ));
}

void main() {
    vec3 rayOrigin = cameraPos;
    vec3 rayDir = normalize(worldPos - cameraPos);
    
    float t = 0.0;
    bool hit = false;
    vec3 hitPos;
    
    for (int i = 0; i < 100; i++) {
        vec3 p = rayOrigin + rayDir * t;
        float d = sdShape(p, 1.0);
        
        if (d < 0.001) {
            hit = true;
            hitPos = p;
            break;
        }
        
        t += d;
        
        if (t > 20.0) break;
    }
    
    if (hit) {
        vec3 normal = calcNormal(hitPos);
        vec3 lightDir = normalize(vec3(1, 1, 1));
        float diffuse = max(dot(normal, lightDir), 0.0);
        vec3 color = vec3(0.2, 0.5, 0.8) * (0.3 + 0.7 * diffuse);
        fragColor = vec4(color, 1.0);
    } else {
        discard;
    }
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

def create_cube_vertices():
    vertices = np.array([
        # Front face
        -1, -1,  1,  0, 0,
         1, -1,  1,  1, 0,
         1,  1,  1,  1, 1,
        -1, -1,  1,  0, 0,
         1,  1,  1,  1, 1,
        -1,  1,  1,  0, 1,
        # Back face
        -1, -1, -1,  1, 0,
        -1,  1, -1,  1, 1,
         1,  1, -1,  0, 1,
        -1, -1, -1,  1, 0,
         1,  1, -1,  0, 1,
         1, -1, -1,  0, 0,
        # Top face
        -1,  1, -1,  0, 1,
        -1,  1,  1,  0, 0,
         1,  1,  1,  1, 0,
        -1,  1, -1,  0, 1,
         1,  1,  1,  1, 0,
         1,  1, -1,  1, 1,
        # Bottom face
        -1, -1, -1,  0, 0,
         1, -1, -1,  1, 0,
         1, -1,  1,  1, 1,
        -1, -1, -1,  0, 0,
         1, -1,  1,  1, 1,
        -1, -1,  1,  0, 1,
        # Right face
         1, -1, -1,  1, 0,
         1,  1, -1,  1, 1,
         1,  1,  1,  0, 1,
         1, -1, -1,  1, 0,
         1,  1,  1,  0, 1,
         1, -1,  1,  0, 0,
        # Left face
        -1, -1, -1,  0, 0,
        -1, -1,  1,  1, 0,
        -1,  1,  1,  1, 1,
        -1, -1, -1,  0, 0,
        -1,  1,  1,  1, 1,
        -1,  1, -1,  0, 1,
    ], dtype=np.float32)
    return vertices

def perspective(fov, aspect, near, far):
    f = 1.0 / np.tan(fov / 2.0)
    return np.array([
        [f/aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
        [0, 0, -1, 0]
    ], dtype=np.float32)

def look_at(eye, center, up):
    f = (center - eye) / np.linalg.norm(center - eye)
    s = np.cross(f, up) / np.linalg.norm(np.cross(f, up))
    u = np.cross(s, f)
    return np.array([
        [s[0], s[1], s[2], -np.dot(s, eye)],
        [u[0], u[1], u[2], -np.dot(u, eye)],
        [-f[0], -f[1], -f[2], np.dot(f, eye)],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def main():
    if not glfw.init():
        sys.exit(1)
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    
    window = glfw.create_window(800, 600, "Cube Texture Shader Test", None, None)
    if not window:
        glfw.terminate()
        sys.exit(1)
    
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    
    program = create_program(VERTEX_SHADER, FRAGMENT_SHADER)
    vertices = create_cube_vertices()
    
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    
    stride = 5 * vertices.itemsize
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * vertices.itemsize))
    
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.1, 0.1, 0.1, 1.0)
    
    mvp_location = glGetUniformLocation(program, "mvp")
    model_location = glGetUniformLocation(program, "model")
    camera_pos_location = glGetUniformLocation(program, "cameraPos")
    
    projection = perspective(np.radians(45), 800/600, 0.1, 100.0)
    model = np.eye(4, dtype=np.float32)
    
    angle = 0.0
    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        angle += 0.01
        eye = np.array([5*cos(angle), 3, 5*sin(angle)], dtype=np.float32)
        view = look_at(eye, np.array([0, 0, 0], dtype=np.float32), np.array([0, 1, 0], dtype=np.float32))
        mvp = projection @ view @ model
        
        glUseProgram(program)
        glUniformMatrix4fv(mvp_location, 1, GL_TRUE, mvp)
        glUniformMatrix4fv(model_location, 1, GL_TRUE, model)
        glUniform3f(camera_pos_location, eye[0], eye[1], eye[2])
        
        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLES, 0, 36)
        
        glfw.swap_buffers(window)
        glfw.poll_events()
    
    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo])
    glDeleteProgram(program)
    glfw.terminate()

if __name__ == "__main__":
    main()

