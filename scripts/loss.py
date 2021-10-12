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

import tensorflow as tf


def ReducedRelativeMSELoss(y_true, y_pred):
    return tf.reduce_mean(RelativeMSELoss(y_true, y_pred)).numpy()


def RelativeMSELoss(y_true, y_pred):
    # Check NaN of network
    tf.debugging.check_numerics(y_pred, 'y_pred')
    y_pred = tf.clip_by_value(y_pred, 0, tf.float32.max)

    true_mean = tf.reduce_mean(y_true, axis=-1)  # Reduce the last channel (RGB)
    true_mean_squared = true_mean * true_mean
    diff_squared = tf.square(y_true - y_pred)
    diff_squared_mean = tf.reduce_mean(diff_squared, axis=-1)
    return diff_squared_mean / (true_mean_squared + 1e-2)


def RelativeL1Loss(y_true, y_pred):
    # Check NaN of network
    tf.debugging.check_numerics(y_pred, 'y_pred')

    true_mean = tf.reduce_mean(y_true, axis=-1)  # Reduce the last channel (RGB)
    diff_abs = tf.abs(y_true - y_pred)
    diff_abs_mean = tf.reduce_mean(diff_abs, axis=-1)
    return diff_abs_mean / (true_mean + 1e-2)


def RelativeL1LogLoss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 0, tf.float32.max)
    return RelativeL1Loss(tf.math.log(1 + y_true), tf.math.log(1 + y_pred))
