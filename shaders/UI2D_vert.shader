#version 330
in vec2 position;
uniform float z_order;
uniform mat3 transform_matrix;
uniform mat3 proj_matrix;


void main()
{
    vec3 rounded_pos = round(transform_matrix*vec3(position, 1.0));
    gl_Position = vec4((proj_matrix*rounded_pos).xy, 1.0/(z_order+0.1), 1.0);
}