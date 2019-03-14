from PyQt5.QtGui import QOpenGLShader, QOpenGLShaderProgram


class ShadersLibrary:
    codes = {
        'simple_texture': [
            """
attribute highp vec4 vertex;
attribute mediump vec4 texCoord;
varying mediump vec4 texc;
uniform mediump mat4 matrix;
void main(void)
{
    gl_Position = matrix * vertex;
    texc = texCoord;
}
                """,
            """
uniform sampler2D texture;
varying mediump vec4 texc;
void main(void)
{
    gl_FragColor = texture2D(texture, texc.st);
}
                """
        ],

        'simple_color_white': [
            """
attribute highp vec4 vertex;
uniform mediump mat4 matrix;
void main(void)
{
    gl_Position = matrix * vertex;
}
                """,
            """
#version 330            
out vec4 color;
void main(void)
{
    color = vec4(0.99, 0.99, 0.99, 1.0);
}
                """
        ]
    }

    cached_shaders = {}

    @staticmethod
    def create_program(name):
        if name not in ShadersLibrary.codes.keys():
            raise KeyError("GL program named "+str(name)+" not found in ShadersLibrary")
        program = ShadersLibrary.cached_shaders.get(name, None)
        if program:
            return program
        vert_code = ShadersLibrary.codes[name][0]
        frag_code = ShadersLibrary.codes[name][1]
        vshader = QOpenGLShader(QOpenGLShader.Vertex)
        vshader.compileSourceCode(vert_code)
        fshader = QOpenGLShader(QOpenGLShader.Fragment)
        fshader.compileSourceCode(frag_code)

        program = QOpenGLShaderProgram()
        program.addShader(vshader)
        program.addShader(fshader)
        program.bindAttributeLocation('vertex', 0)
        if(name != 'simple_color_white'):
            program.bindAttributeLocation('texCoord', 1)
        if not program.link():
            raise RuntimeError("Could not compile shader:"+str(name)+"\n"+program.log())

        program.setUniformValue('texture', 0)
        program.enableAttributeArray(0)
        program.enableAttributeArray(1)
        ShadersLibrary.cached_shaders[name] = program
        return program

