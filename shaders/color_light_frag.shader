#version 330
in vec3 frag_normal;
in vec3 frag_position;
uniform mediump vec3 light_position;
uniform highp vec4 material_color;

out vec4 color;
void main(void)
{
   float ndotl = max(dot(frag_normal, normalize(light_position - frag_position)), 0.0);
   vec4 L = vec4(ndotl * material_color.xyz, 1.0);
   color = clamp(L, 0.0, 1.0);
   //color = vec4(frag_normal,1.0);
}