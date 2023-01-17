import numpy as np
import random
import os

import json
import sys

import time

from vae.vae import ConvVAE
from rnn.rnn import hps_sample, MDNRNN, rnn_init_state, rnn_next_state, rnn_output, rnn_output_size

render_mode = True

# controls whether we concatenate (z, c, h), etc for features used for car.
MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3  # extra hidden later
MODE_ZH = 4

EXP_MODE = MODE_ZH


def make_model(load_model=True):
    # can be extended in the future.
    model = Model(load_model=load_model)
    return model


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def clip(x, lo=0.0, hi=1.0):
    return np.minimum(np.maximum(x, lo), hi)


def passthru(x):
    return x


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sample(p):
    return np.argmax(np.random.multinomial(1, p))

def main2():
    md = MDNRNN(hps_sample, gpu_mode=False, reuse=True)
    md.load_json('rnn/rnn.json')
    state = rnn_init_state(md)

    raw_data = np.load( "series/series.npz")
    action=raw_data["action"]
    mu=raw_data["mu"]
    logvar=raw_data["logvar"]

    z = mu + np.exp(logvar / 2.0)
    z = z[0][0]
    action=action[0][0]

    state = rnn_next_state(md, z, action, state)

    print("end")



if __name__ == "__main__":
    main2()
