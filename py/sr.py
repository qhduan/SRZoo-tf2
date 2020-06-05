#!/usr/bin/env python
# coding: utf-8

import sys
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np


def read_img(path):
    img = Image.open(path)
    arr = np.array(img)
    tfarr = tf.constant([arr])
    tfarr = tf.cast(tfarr, tf.float32)
    return tfarr


def predict(m, img, channel_first=True, pixel_max_one=False):

    if channel_first:
        img = tf.transpose(img, [0, 3, 1, 2])
    if pixel_max_one:
        img = img / 255.0

    pred = m(img)
    out = pred['out']

    if channel_first:
        out = tf.transpose(out, [0, 2, 3, 1])
    if pixel_max_one:
        out = out * 255.0

    out = out[0].numpy()
    img = Image.fromarray(np.round(np.clip(out, 0, 255)).astype(np.uint8), "RGB")
    return img


model_path = sys.argv[1]
img_path = sys.argv[2]
output_path = sys.argv[3]

channel_first = True
pixel_max_one = False

if 'carn' in model_path:
    pixel_max_one = True
if 'dbpn' in model_path:
    pixel_max_one = True
if 'esrgan' in model_path:
    pixel_max_one = True
if 'frsr' in model_path:
    pixel_max_one = True
if 'natsr' in model_path:
    pixel_max_one = True
if 'rrdb' in model_path:
    pixel_max_one = True

if '4pp_eusr' in model_path:
    channel_first = False
if 'eusr' in model_path:
    channel_first = False
if 'frsr' in model_path:
    channel_first = False
if 'natsr' in model_path:
    channel_first = False

print(model_path, img_path, output_path)
m = hub.KerasLayer(model_path, signature='serving_default', signature_outputs_as_dict=True)

img = read_img(img_path)
uimg = predict(m, img, channel_first=channel_first, pixel_max_one=pixel_max_one)
uimg.save(output_path)

