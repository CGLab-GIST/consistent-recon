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

import time
import re
from functools import partial
from sys import exit
import multiprocessing
from multiprocessing import Pool
import parmap
import exr
from utilities import printFunc, samplePatchesStrided
import numpy as np
import tensorflow as tf
import os
import random
from functools import partial
import gc


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def load_color(directory, endMatch, train, config):
    filenames = []
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        if path.endswith(endMatch):
            filenames.append(path)

    filenames = natural_sort(filenames) # Natural sort

    # Shuffle the training images
    if train:
        random.seed(time.time())
        random.shuffle(filenames)
    
    if len(filenames) == 0:
        printFunc(directory, 'is empty!')
        exit()

    return filenames


def load_reference(color_filenames, config):
    color_dataset_dir = re.search(
        '.*' + config["datasetPrefix"] + '.*\/', color_filenames[0]).group(0)
    ref_dataset_dir = color_dataset_dir.replace(re.search(
        config["datasetPrefix"] +
        '.*\/', color_filenames
        [0]).group(0),
        'dataset_references/')

    ref_candidates = []
    for file in os.listdir(ref_dataset_dir):
        path = os.path.join(ref_dataset_dir, file)
        if path.endswith('.exr'):
            ref_candidates.append(path)

    ref_filenames = []
    for color_filename in color_filenames:
        color_dataset_dir = re.search(
            '.*' + config["datasetPrefix"] + '.*\/', color_filename).group(0)
        cname = color_filename.split(color_dataset_dir)[-1]
        color_scenename = cname.split('_rand')[0]

        found = False
        for ref_filename in ref_candidates:
            ref_scenename = ref_filename.split('/')[-1].split('_rand')[0]
            if color_scenename == ref_scenename:
                found = True
                ref_filenames.append(ref_filename)
                break

        if not found:
            printFunc('Cannot find reference for', color_filename)
            exit()

    save_exr_to_npy(ref_filenames)
    ref_filenames = list(
        map(lambda x: x.replace('.exr', '.npy'), ref_filenames))
    
    if len(ref_filenames) == 0:
        printFunc(ref_dataset_dir, 'is empty!')
        exit()

    return ref_filenames


def check_paris(color_filenames, ref_filenames):
    for (c, r) in zip(color_filenames, ref_filenames):
        c = c.split('_rand')[0].split('/')[-1]
        r = r.split('_rand')[0].split('/')[-1]
        if c != r:
            print('Wrong pair!', c, r)
            exit(-1)


def load_tf(path):
    def load_numpy(color_path, ref_path):
        sceneName = color_path.numpy().decode('utf-8').split('/')[-1].split('.')[0]
        return sceneName, np.load(color_path.numpy()), np.load(ref_path.numpy())
    return tf.py_function(load_numpy, inp=[path[0], path[1]], Tout=[tf.string, tf.float32, tf.float32])


def patch_generator(config, paths):
    for idx, (img_path, ref_path) in enumerate(paths.tolist()):
        img, ref = np.load(img_path), np.load(ref_path)
        patch_indices = samplePatchesStrided(img.shape[:2], config['patchSize'], config['patchStride'])
        for pos in patch_indices:
            yield img[pos[1]:pos[1] + config['patchSize'], pos[0]:pos[0] + config['patchSize']], ref[pos[1]:pos[1] + config['patchSize'], pos[0]:pos[0] + config['patchSize']]


def load(config, train, preprocessor=None):
    color_filenames = []
    if train:
        for scene in config['trainScenes']:
            directory = os.path.join(config['trainDatasetDirectory'], scene)
            # Collect filenames
            color_filenames.extend(load_color(directory, "passes.exr", train, config))
    else:
        for scene in config['testScenes']:
            directory = os.path.join(config['testDatasetDirectory'], scene)
            # Collect filenames
            color_filenames.extend(load_color(directory, "passes.exr", train, config))
    # Load exr file and save it to npy for buffer images
    save_exr_to_npy(color_filenames, config["channelNames"])
    color_filenames = list(map(lambda x: x.replace('.exr', '.npy'), color_filenames))
    # Load reference files
    ref_filenames = load_reference(color_filenames, config)
    # Check parity of collected names
    check_paris(color_filenames, ref_filenames)
    # Combine two lists
    filenames = np.stack([color_filenames, ref_filenames], axis=1)

    numData = len(filenames)
    if train:
        printFunc('[%s] num train data: %d' % (config['trainDatasetDirectory'], numData))
    else:
        printFunc('[%s] num test data: %d' % (config['testDatasetDirectory'], numData))

    if train:
        dataset = tf.data.Dataset.from_generator(partial(patch_generator, config, filenames), 
            output_signature=(
                tf.TensorSpec(shape=(None, None, config['numChannels']), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32)
            ))
        dataset = dataset.shuffle(buffer_size=config['shuffleBufferSize'], reshuffle_each_iteration=True)
        dataset = dataset.batch(config['patchBatch'])
    else:
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(load_tf)
        dataset = dataset.batch(1)

    # Pre-process the input
    # if preprocessor:
    #     dataset = preprocessor(dataset)

    return dataset.prefetch(tf.data.AUTOTUNE)


def convert_exr_to_npy(filename, channels):
    new_filename = filename.replace('exr', 'npy')
    if os.path.isfile(new_filename):
        return
    # printFunc('convert', filename, 'to', new_filename)
    data = exr.read_all(filename)
    if len(channels) == 1:
        accm = data[channels[0]]
    else:
        # Append channels into a tensor
        accm = data[channels[0]]
        for i in range(1, len(channels)):
            c = channels[i]
            accm = np.dstack([accm, data[c]])

    if np.isnan(accm).any():
        printFunc('There is NaN in', new_filename, 'Set it to zero for training.')
        accm = np.nan_to_num(accm, copy=False)
    if np.isposinf(accm).any() or np.isneginf(accm).any():
        printFunc("There is INF in", new_filename, 'Set it to zero for training.')
        accm[accm == np.inf] = 0
        accm[accm == -np.inf] = 0

    np.save(new_filename, accm)


def save_exr_to_npy(filenames, channels=['default']):
    filenames = list(dict.fromkeys(filenames))
    # Remove Component Name and Merge Channels (RGB to single channel)
    channels = list(map(lambda x: x.replace('.R', '').replace('.G', '').replace(
        '.B', '').replace('.A', '').replace('.Y', ''), channels))
    channels = list(dict.fromkeys(channels))
    # print('cpu count:', multiprocessing.cpu_count())
    parmap.map(partial(convert_exr_to_npy, channels=channels), filenames, pm_pbar=True, pm_processes=multiprocessing.cpu_count())
