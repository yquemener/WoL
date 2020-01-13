#version 330
uniform mediump vec4 material_color;
out vec4 color;
void main(void)
{
    color = material_color;
}