attribute highp vec4 vertex;
uniform mediump mat4 matrix;
void main(void)
{
    gl_Position = matrix * vertex;
}