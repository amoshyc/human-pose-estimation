import math
import pathlib
from tqdm import tqdm

from keras.layers import *
from keras.models import Model, Sequential
from keras.optimizers import *


def _make_block(c, k=3, p='same', s=1, act='relu'):
    block = Sequential()
    block.add(Conv2D(c), k, s, padding=p)
    block.add(BatchNormalization())
    block.add(Activation(act))
    return block


def _make_hg(c):
    inp = Input(shape=shape)
    z1 = _make_block(c, k=3)(inp)
    z2 = _make_block(c, k=3)(MaxPooling2D(padding='same')(z1, 2))
    z3 = _make_block(c, k=3)(MaxPooling2D(padding='same')(z1, 2))
    z4 = _make_block(c, k=3)(MaxPooling2D(padding='same')(z1, 2))
    z = MaxPooling2D(padding='same')(z4)

    z1 = _make_block(c, k=3)(z1)
    z2 = _make_block(c, k=3)(z2)
    z3 = _make_block(c, k=3)(z3)
    z4 = _make_block(c, k=3)(z4)
    z = Sequential(
        [_make_block(c, k=3),
         _make_block(c, k=3),
         _make_block(c, k=3)])(z)

    z4 = _make_block(c, k=3)(Add()([z4, UpSampling2D(z)]))
    z3 = _make_block(c, k=3)(Add()([z3, UpSampling2D(z4)]))
    z2 = _make_block(c, k=3)(Add()([z2, UpSampling2D(z3)]))
    z1 = _make_block(c, k=3)(Add()([z1, UpSampling2D(z2)]))

    return Model(inputs=inp, outputs=z1)


def _make_ae():
    inp = Input(shape=(256, 256, 3))
    x = _make_block(32, k=7, s=2)(inp)
    x = _make_hg(32, shape=(128, 128, 32))(x)
    x = _make_block(32, k=1, act='linear')(x)
    return Model(inputs=inp, outputs=x)


if __name__ == '__main__':
    model = _make_ae()
    model.summary()