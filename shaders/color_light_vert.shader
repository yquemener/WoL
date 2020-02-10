#version 330 core

layout(location = 0) in vec3 vertex;
layout(location = 1) in vec3 texture_uv;
layout(location = 2) in vec3 normal;
uniform mediump mat4 mvp;
uniform mediump mat4 matmodel;
varying vec3 frag_normal;
varying vec3 frag_position;
void main(void)
{
    mat3 normalMatrix = transpose(inverse(mat3(matmodel)));
    frag_normal = normalize(normalMatrix*normal);
    //frag_normal = gl_Normal;
    frag_position = (matmodel * vec4(vertex, 1.0)).xyz;
    gl_Position = mvp * vec4(vertex, 1.0);
}