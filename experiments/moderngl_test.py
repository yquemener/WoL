'''
    Renders a floating, oscillating, 3d islan with lights
'''

import os

import moderngl
import numpy as np
from objloader import obj as Obj
from PIL import Image
from pyrr import Matrix44

import data
from window import Example, run_example


class ColorsAndTexture(Example):
    gl_version = (3, 3)
    title = "Colors and Textures"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 Mvp;
                in vec3 in_vert;
                in vec3 in_norm;
                in vec2 in_text;
                out vec3 v_vert;
                out vec3 v_norm;
                out vec2 v_text;
                void main() {
                    gl_Position = Mvp * vec4(in_vert, 1.0);
                    v_vert = in_vert;
                    v_norm = in_norm;
                    v_text = in_text;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform vec3 Light;
                uniform vec3 Color;
                uniform bool UseTexture;
                uniform sampler2D Texture;
                in vec3 v_vert;
                in vec3 v_norm;
                in vec2 v_text;
                out vec4 f_color;
                void main() {
                    float lum = clamp(dot(normalize(Light - v_vert), normalize(v_norm)), 0.0, 1.0) * 0.8 + 0.2;
                    if (UseTexture) {
                        f_color = vec4(texture(Texture, v_text).rgb * lum, 1.0);
                    } else {
                        f_color = vec4(Color * lum, 1.0);
                    }
                }
            ''',
        )

        self.mvp = self.prog['Mvp']
        self.light = self.prog['Light']
        self.color = self.prog['Color']
        self.use_texture = self.prog['UseTexture']

        self.objects = {}

        for name in ['ground', 'grass', 'billboard', 'billboard-holder', 'billboard-image']:
            obj = Obj.open(data.find('scene-1-%s.obj' % name))
            vbo = self.ctx.buffer(obj.pack('vx vy vz nx ny nz tx ty'))
            vao = self.ctx.simple_vertex_array(self.prog, vbo, 'in_vert', 'in_norm', 'in_text')
            self.objects[name] = vao

        img = Image.open(data.find('infographic-1.jpg')).transpose(Image.FLIP_TOP_BOTTOM).convert('RGB')
        self.texture = self.ctx.texture(img.size, 3, img.tobytes())
        self.texture.build_mipmaps()

    def render(self, time: float, frame_time: float):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)
        lookat = Matrix44.look_at(
            (47.697, -8.147, 24.498),
            (0.0, 0.0, 8.0),
            (0.0, 0.0, 1.0),
        )

        rotate = Matrix44.from_z_rotation(np.sin(time) * 0.5 + 0.2)

        self.use_texture.value = False

        self.light.value = (67.69, -8.14, 52.49)
        self.mvp.write((proj * lookat * rotate).astype('f4').tobytes())

        self.color.value = (0.67, 0.49, 0.29)
        self.objects['ground'].render()

        self.color.value = (0.46, 0.67, 0.29)
        self.objects['grass'].render()

        self.color.value = (1.0, 1.0, 1.0)
        self.objects['billboard'].render()

        self.color.value = (0.2, 0.2, 0.2)
        self.objects['billboard-holder'].render()

        self.use_texture.value = True
        self.texture.use()

        self.objects['billboard-image'].render()


if __name__ == '__main__':
    run_example(ColorsAndTexture)