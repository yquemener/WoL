#version 430
in mediump vec4 uv;
out vec4 col;
layout(std430, binding=0) readonly buffer Buf { float data[]; };
uniform int w1;
uniform int w2;
uniform int w3;
uniform float value_scale;
void main() { 
    int x = int(uv.x * (w1-0.1));
    int y = int(uv.y * (w2-0.1));
    int z = int(uv.z * (w3-0.1));
    int i = (z*w2*w1 + y*w1 + x);
    col = vec4(data[i]*value_scale, data[i]*value_scale, data[i]*value_scale, 1.0); 
}