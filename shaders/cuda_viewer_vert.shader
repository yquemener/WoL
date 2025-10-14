#version 430
in vec4 pos;
in vec4 texc;
out vec4 uv;
uniform mat4 matrix;
void main() { gl_Position = matrix * pos; uv = texc; }
