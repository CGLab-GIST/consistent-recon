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
from loss import RelativeL1LogLoss, RelativeMSELoss, RelativeL1Loss

# tf.config.experimental_run_functions_eagerly(False)
# tf.config.run_functions_eagerly(False)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


######################
## Global Variables ##
######################
config = {}
config["timezone"] = "Asia/Seoul"
# Name of checkpoints
config["title"] = "v9"

# Mode selection [TRAINING, TEST]
config["mode"] = "TEST"
config["loadEpoch"] = "100"

# System memory related params
config["patchBatch"] = 8 # Number of patches per batch (and per replica)
print('Batch size for single-GPU:', config["patchBatch"])
config['shuffleBufferSize'] = config["patchBatch"] * 2

# Training
config['epochs'] = 100

# Network
config["loss"] = RelativeL1LogLoss
config["convSize"] = 5
config["convActivation"] = 'relu'
config["numFilters"] = 50
config["kernelSize"] = 19
config["kernelArea"] = config["kernelSize"] * config["kernelSize"]
config["numCandidates"] = 5
config["numOutput"] = (config["numCandidates"] - 1) * config["kernelArea"]
config["patchSize"] = 64
config["patchStride"] = int(config["patchSize"] * 1.5)


##############
## Dataset  ##
##############
config["datasetPrefix"] = 'dataset'
config["trainDatasetDirectory"] = "dataset_train"
config["testDatasetDirectory"] = "dataset_test"
config["trainScenes"] = ['bathroom', 'bathroom2', 'box', 'classroom', 'living-room', 'living-room-2', 'spaceship', 'sponza', 'staircase', 'torus', 'veach-lamp', 'glass']
config["testScenes"] = ['bookshelf', 'breakfast-room', 'pool', 'water']


#############################
## Structure of Input File ##
#############################
config["channelNames"] = [
    "00_albedo.R", "00_albedo.G", "00_albedo.B",        #
    "01_normal.R", "01_normal.G", "01_normal.B",        #
    "02_depth.Y",                                       #
    "03_iteration.Y",                                   #
]

channelPos = int(config["channelNames"][-1].split("_")[0]) + 1

pos = 0
for i in range(config["numCandidates"]):
    config["channelNames"].append(
        "%02d_variance%02d.Y" % (channelPos, pos))
    channelPos += 1
    pos += 1

pos = 0
for i in range(config["numCandidates"]):
    config["channelNames"].append(
        "%02d_radius%02d.Y" % (channelPos, pos))
    channelPos += 1
    pos += 1

pos = 0
for i in range(config["numCandidates"]):
    config["channelNames"].append(
        "%02d_candidate%02d.R" % (channelPos, pos))
    config["channelNames"].append(
        "%02d_candidate%02d.G" % (channelPos, pos))
    config["channelNames"].append(
        "%02d_candidate%02d.B" % (channelPos, pos))
    channelPos += 1
    pos += 1

# 33 = 7 (albedo, normal, depth) + 1 (iter) + 5 (candidate variance) + 5 (candidate radius) + 5 (numCandidates) * 3 (RGB)
config["numChannels"] = len(config["channelNames"])

config["ALBEDO_POS"] = 0
config["NORMAL_POS"] = 3
config["DEPTH_POS"] = 6
config["ITERATION_POS"] = 7
config["VARIANCE_POS"] = 8
config["RADIUS_POS"] = config["VARIANCE_POS"] + config['numCandidates']
config["CANDIDATE_POS"] = config["RADIUS_POS"] + config['numCandidates']

