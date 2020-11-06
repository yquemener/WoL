from PyQt5.QtGui import QOpenGLShader, QOpenGLShaderProgram


class ShadersLibrary:
    codes = {
        'simple_texture': [open("shaders/texture_vert.shader").read(),
                           open("shaders/texture_frag.shader").read()],

        'simple_color_white': [open("shaders/simple_vert.shader").read(),
                               open("shaders/white_frag.shader").read()],
        'simple_color': [open("shaders/simple_vert.shader").read(),
                         open("shaders/color_frag.shader").read()],

        'simple_lighting': [open("shaders/color_light_vert.shader").read(),
                            open("shaders/color_light_frag.shader").read()],

        'hud_2d': [open("shaders/UI2D_vert.shader").read(),
                   open("shaders/color_frag.shader").read()],

        'hud_2d_tex': [open("shaders/UI2D_vert.shader").read(),
                       open("shaders/texture_frag.shader").read()],

        'wireframe': [open("shaders/simple_vert.shader").read(),
                      open("shaders/color_frag.shader").read()]
    }

    cached_shaders = {}

    @staticmethod
    def create_program(name):
        if name not in ShadersLibrary.codes.keys():
            raise KeyError("GL program named " + str(name) + " not found in ShadersLibrary")
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
        if name == 'simple_texture':
            program.bindAttributeLocation('texCoord', 1)

        if not program.link():
            raise RuntimeError("Could not compile shader:" + str(name) + "\n" + program.log())

        program.setUniformValue('texture', 0)
        program.enableAttributeArray(0)
        program.enableAttributeArray(1)
        ShadersLibrary.cached_shaders[name] = program
        return program
