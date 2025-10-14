#version 430
in mediump vec4 uv;
out vec4 col;
layout(std430, binding=0) readonly buffer Buf { float data[]; };
uniform int w;
uniform float scale;
void main() { 
    int x = int(uv.x * w);
    int y = int(uv.y * w);
    int i = (y * w  + x);
    col = vec4(data[i]*scale, data[i]*scale, data[i]*scale, 1.0); 
}