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

import os
import psutil
import time
from logger import Logger

os.environ['MPLCONFIGDIR'] = '/tmp'  # 0, 1, 2, 3s
import numpy as np

# Disable Tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0, 1, 2, 3s
import tensorflow as tf

import exr
import loader
from layers import MainNet
from utilities import printFunc
from config import config, gpus
from loss import RelativeMSELoss


def parseInput(config, input):
    # input: (B, H, W, numChannels)
    # G-buffer (7)
    albedo = input[:, :, :, config["ALBEDO_POS"]:config["ALBEDO_POS"] + 3]
    normal = input[:, :, :, config["NORMAL_POS"]:config["NORMAL_POS"] + 3]
    depth = input[:, :, :, config["DEPTH_POS"]:config["DEPTH_POS"] + 1]
    # Iteration (1)
    iter = tf.cast(input[:, :, :, config["ITERATION_POS"]:config["ITERATION_POS"] + 1], tf.float32)
    # Candidate Variance (5)
    var = tf.cast(
        input[:, :, :, config["VARIANCE_POS"]: config["VARIANCE_POS"] +
              config['numCandidates']],
        tf.float32)
    # Candidate Radius (5)
    radius = tf.cast(
        input[:, :, :, config["RADIUS_POS"]: config["RADIUS_POS"] +
              config["numCandidates"]],
        tf.float32)
    # Candidate Colors (3 * 5)
    candidates = input[:, :, :, config["CANDIDATE_POS"]:config["CANDIDATE_POS"] +
                       config['numCandidates'] * 3]  # 3 for RGB

    return (albedo, normal, depth, iter, var, radius, candidates)


def buildModel():
    # Update the global variable
    global model

    ####################
    ###### Input #######
    ####################
    # Input shape can be (patchSize, patchSize) or (height, width) at inference
    input = tf.keras.Input(shape=(None, None, config['numChannels']), name="Input")

    ####################
    ###### Model #######
    ####################
    output, weights = MainNet(config, input)
    model = tf.keras.Model(inputs=[input], outputs=[output, weights], name="my_model")

    ####################
    ###### Model #######
    ####################
    sum = 0
    for v in model.trainable_variables:
        print(v.name, v.shape)
        sum += tf.math.reduce_prod(v.shape)
    print('\tNumber of weights:', f"{sum.numpy():,}")

    return model


def evaluateDataset(dataset, name, epoch=0, isTrainDataset=False, numOutputBatch=1, batchStride=1, loss=None, save=True):
    cnt = 0

    # Iterate over batchs or images
    start = time.time()
    for (batchIdx, element) in enumerate(dataset):
        if batchIdx % batchStride != 0:
            continue

        if isTrainDataset:
            input, ref = element
        else:
            sceneName, input, ref = element
            sceneName = sceneName.numpy()[0].decode('utf-8')

        (_, _, _, _, _, radius, candid) = parseInput(config, input)
        # with tf.device("/cpu:0"): # Uncomment this line if the OOM occurs
        (out_color, weights) = model(input, training=False)

        # Iterate over each image
        for sampleIdx, img in enumerate(out_color):
            if save:
                centerCandid = candid[:, :, :, 6:9]
                ours_loss = tf.reduce_mean(config['loss'](ref, img)).numpy()
                ours_rmse = tf.reduce_mean(RelativeMSELoss(ref, img)).numpy()
                cent_loss = tf.reduce_mean(config['loss'](ref, centerCandid)).numpy()
                cent_rmse = tf.reduce_mean(RelativeMSELoss(ref, centerCandid)).numpy()

                if isTrainDataset:
                    filename = "output/%s_epoch%04d_%s_batch%03d_img%03d.exr" % (
                        config["title"], epoch, name, batchIdx, sampleIdx)
                else:
                    filename = "output/%s_epoch%04d_%s%f.exr" % (
                        sceneName, epoch, 'rMSE', ours_rmse)

                # Write output color
                exr.write(filename.replace('.exr', '_color.exr'), img.numpy())

                # # Write weight images
                # output = []
                # for i in range(config['numCandidates'] - 1):
                #     start = i * config['kernelArea']
                #     end = (i + 1) * config['kernelArea']
                #     output.append(np.sum(weights[sampleIdx][:,:,start:end].numpy(), axis=-1))
                # output = np.stack(output, axis=-1)
                # exr.write('output/%s_weights.exr' % sceneName, output, ['0', '1', '2', '3'])

                # # Evaluate the input images
                # logfile_ours.write(str(ours_rmse))
                # for i in range(config['numCandidates']):
                #     sppm_rmse = tf.reduce_mean(RelativeMSELoss(ref, candid[:, :, :, i*3:i*3+3])).numpy()
                #     logfile_ours.write(' ' + str(sppm_rmse))
                # logfile_ours.write('\n')
                # logfile_ours.flush()

                # Write independent (least radius)
                # exr.write(filename.replace('.exr', '_independent.exr'), candid[sampleIdx,:,:,3*(config['numCandidates']-1):].numpy())

        cnt += 1
        if numOutputBatch >= 0 and cnt >= numOutputBatch:
            break
    printFunc("\tTook", time.time() - start, "s for evaluateDataset()")


if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = buildModel()

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, )

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        checkpoint_dir = "checkpoints"
        checkpoint_path = checkpoint_dir + "/cp-%s" % config['title']

        if config['mode'] == "TEST":
            dataset_test = loader.load(config, False, None)

            # Load weights
            checkpoint.restore(checkpoint_path + "-" + config["loadEpoch"])
            evaluateDataset(dataset_test, 'final', epoch=0, numOutputBatch=-1, batchStride=1)

        elif config['mode'] == "TRAINING":
            dataset = loader.load(config, True, None)
            dataset_test = loader.load(config, False, None)
            # Just return dataset to make it as distributed
            # The patchBatch will be allocated for each replica
            # WARNING: `experimental_distribute_dataset` causes massive memory leak
            dataset = strategy.distribute_datasets_from_function(lambda context: dataset)

            # Fine-tuning or re-training
            checkpoint.restore(checkpoint_path + "-" + config["loadEpoch"])

            # Train
            curr_epoch = 0

            start = time.time()

            def compute_loss(labels, predictions):
                per_example_loss = config['loss'](labels, predictions)
                # reduce_mean except the first dimension (batch dim)
                per_example_loss = tf.reduce_mean(per_example_loss, axis=[1, 2])
                return tf.nn.compute_average_loss(
                    per_example_loss, global_batch_size=len(gpus) * config['patchBatch'])

            def train_step(model, inputs):
                images, refs = inputs

                with tf.GradientTape() as tape:
                    predictions, weights = model(images, training=True)
                    loss = compute_loss(refs, predictions)

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                return loss

            # `run` replicates the provided computation and runs it
            # with the distributed input.
            @tf.function
            def distributed_train_step(model, dataset_inputs):
                per_replica_losses = strategy.run(train_step, args=(model, dataset_inputs))
                return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

            process = psutil.Process(os.getpid())
            logger = Logger('log.txt', 'a+', config['timezone'])

            globalStart = time.time()
            for epoch in range(1, config['epochs'] + 1):
                total_loss = 0.0
                batch = 0

                start = time.time()
                for x in dataset:
                    total_loss += distributed_train_step(model, x)
                    batch += 1
                    if batch == 1 or batch % 10 == 0:
                        mem = int(process.memory_info().rss / 1024 / 1024)
                        print('[ Denoising ] [ Epoch %03d / %03d ] [ Batch %04d / %04d ] Train loss: %.6f, Memory: %d MB' %
                              (epoch, config['epochs'], batch, 0, total_loss.numpy() / batch, mem))
                train_loss = total_loss / batch

                # Save checkpoint
                if epoch % 1 == 0:
                    checkpoint.save(checkpoint_path)

                duration = time.time() - start
                logger.add_loss(train_loss.numpy(), epoch, title='Multi', time=duration)

            evaluateDataset(dataset_test, 'final', epoch=0,
                                          numOutputBatch=-1, batchStride=1, save=True)

            print('Training took', '%.2fs' % (time.time() - globalStart),
                  'for %d epochs.' % (config['epochs']))
