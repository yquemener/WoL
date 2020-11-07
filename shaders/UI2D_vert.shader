attribute highp vec4 vertex;
attribute mediump vec4 texCoord;
varying mediump vec4 texc;

uniform mat4 matrix;
uniform float z_order;


void main()
{

//    vec4 rounded_pos = round(vec4(vertex.xy, 1.0, 1.0));
    //gl_Position = vec4((matrix*rounded_pos).xy, 1.0/(z_order+0.1), 1.0);
//    gl_Position = vec4(vertex.x/500-1, vertex.y/500-1, 0.9, 1.0);
    vec4 transfomed = matrix*vec4(vertex.xy, z_order, 1.0);
    gl_Position = vec4(transfomed.x, transfomed.y, 0.9, 1.0);
    texc = texCoord;
}