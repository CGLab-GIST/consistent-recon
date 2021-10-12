""" 
BSD 2-Clause License

Copyright (c) 2021, CGLAB
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
"""

import copy
import tensorflow as tf
from tensorflow.keras import layers, activations
from tensorflow.keras.initializers import GlorotUniform, zeros


_module = tf.load_op_library('./_weightaverage_ops.so')

@tf.RegisterGradient("WeightedAverage")
def _weighted_average_grad(op, grad):
    image = op.inputs[0]
    weights = op.inputs[1]
    grads = _module.weighted_average_grad(grad, image, weights)
    grads = tf.clip_by_value(grads, -1000000, 1000000)
    return [None, grads]

weighted_average = _module.weighted_average


def kernelPredictingWeights(z):
    # z: (B, H, W, kernelArea)

    # [-inf, 0], for numerical stability
    w = z - tf.reduce_max(z)
    # [0, 1]
    w = activations.softmax(w)
    return w


def conv2d(x, config):
    return layers.Conv2D(filters=config['numFilters'],
                         kernel_size=(config['convSize'], config['convSize']),
                         activation=config["convActivation"],
                         padding='same',
                         strides=(1, 1),
                         kernel_initializer=GlorotUniform(),
                         bias_initializer=zeros())(x)


def conv2d_last(x, config):
    return layers.Conv2D(
        filters=config['numOutput'],
        kernel_size=(config['convSize'], config['convSize']),
        padding='same', strides=(1, 1),
        kernel_initializer=GlorotUniform(),
        bias_initializer=zeros())(x)  # Constant to make initial radius to be 1


def ConvolutionNet(config, x):
    # x: (B, H, W, numInputChannels)

    # x: (Batch, H, W, 100)
    x = conv2d(x, config)

    for i in range(8):
        # x: (Batch, H, W, 100)
        x = conv2d(x, config)

    # x: (Batch, H, W, numOutput)
    x = conv2d_last(x, config)
    return x


def MainNet(config, input):
    # input: (B, H, W, numChannels)
    N = input[:, :, :, config['ITERATION_POS']:config['ITERATION_POS'] + 1]
    albedo = input[:, :, :, config['ALBEDO_POS']:config['ALBEDO_POS'] + 3]
    normal = input[:, :, :, config['NORMAL_POS']:config['NORMAL_POS'] + 3]
    depth = input[:, :, :, config['DEPTH_POS']:config['DEPTH_POS'] + 1]
    var = input[:, :, :, config['VARIANCE_POS']:config['VARIANCE_POS'] + config['numCandidates']]
    candidates = input[:, :, :, config['CANDIDATE_POS']
        :config['CANDIDATE_POS'] + 3 * config['numCandidates']]

    # x: (B, H, W, numInputChannels)
    x = tf.concat([albedo, normal, depth, var, candidates], axis=3)

    # x: (B, H, W, numOutput)
    x = ConvolutionNet(config, x)

    # (B, H, W, kernelArea * (numCandidates-1))
    denoisingWeights = activations.relu(x) + 1e-4  # to prevent all zero
    denoisingWeights = denoisingWeights / tf.reduce_sum(denoisingWeights, axis=-1, keepdims=True)

    lastCandidIdx = 3 * (config['numCandidates'] - 1)
    # (B, H, W, 3): the candidate with least radius 
    yi = candidates[:, :, :, lastCandidIdx:lastCandidIdx + 3]
    # (B, H, W, 3)
    output = tf.zeros_like(albedo)

    for i in range(config['numCandidates'] - 1):
        start = i * config['kernelArea']
        end = (i + 1) * config['kernelArea']

        # (B, H, W, kernelArea)
        wb = denoisingWeights[:, :, :, start:end]
        # (B, H, W, 3)
        zb = candidates[:, :, :, i * 3:i * 3 + 3]

        # (B, H, W, 3)
        denoised = weighted_average(yi - zb, wb)
        output += denoised

        # (B, H, W, 1)
        sumWeights = tf.reduce_sum(wb, axis=-1, keepdims=True)
        output += zb * sumWeights

    return output, denoisingWeights
