attribute highp vec4 vertex;
uniform mediump mat4 mvp;
uniform mediump mat4 matmodel;
varying vec3 frag_normal;
varying vec3 frag_position;
void main(void)
{
    mat3 normalMatrix = transpose(inverse(mat3(matmodel)));
    frag_normal = normalize(normalMatrix*gl_Normal);
    //frag_normal = gl_Normal;
    frag_position = (matmodel * vertex).xyz;
    gl_Position = mvp * vertex;
}