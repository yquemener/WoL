attribute highp vec4 vertex;
attribute mediump vec4 texCoord;
varying mediump vec4 texc;

uniform mat4 matrix;
uniform float z_order;


void main()
{
    vec4 rounded_pos = round(vec4(vertex.xy, 1.0, 1.0));
    gl_Position = vec4((matrix*rounded_pos).xy, 1.0/(z_order+0.1), 1.0);
    texc = texCoord;
}