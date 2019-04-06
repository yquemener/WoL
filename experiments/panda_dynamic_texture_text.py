from __future__ import division

import numpy as np

from panda3d.core import Texture, CardMaker
from direct.showbase.ShowBase import ShowBase

h, w = 480, 640
cap = None

# setup panda3d scripting env (render, taskMgr, camera etc)
base = ShowBase()

# set up a texture for (h by w) 8 bit gray scale image
tex = Texture()
tex.setup2dTexture(h, w, Texture.T_unsigned_byte, Texture.F_luminance)

# set up a card to apply the numpy texture
cm = CardMaker('card')
card = render.attachNewNode(cm.generate())
card.setPos(-0.5, 1.5, -w / h / 2)  # bring it to center, put it in front of camera
card.setScale(1, 1, w / h)  # card is square, rescale to the original image aspect


def updateTex(task):
    frame = np.random.randint(0, 255, (h, w, 3)).astype(np.uint8)

    buf = frame[:, :, 0].T.tostring()  # slice RGB to gray scale, transpose 90 degree, convert to text buffer
    tex.setRamImage(buf)  # overwriting the memory with new buffer
    card.setTexture(tex)  # now apply it to the card

    return task.cont


taskMgr.add(updateTex, 'video frame update')

base.run()
