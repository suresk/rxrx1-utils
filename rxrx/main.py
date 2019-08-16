# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train a ResNet-50 model on RxRx1 on TPU.

Original file:
    https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_main.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import os
import time
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.python.estimator import estimator

from .model import model_fn

from rxrx import input as rxinput

DEFAULT_INPUT_FN_PARAMS = {
    'tfrecord_dataset_buffer_size': 256,
    'tfrecord_dataset_num_parallel_reads': None,
    'parallel_interleave_cycle_length': 32,
    'parallel_interleave_block_length': 1,
    'parallel_interleave_buffer_output_elements': None,
    'parallel_interleave_prefetch_input_elements': None,
    'map_and_batch_num_parallel_calls': 128,
    'transpose_num_parallel_calls': 128,
    'prefetch_buffer_size': tf.contrib.data.AUTOTUNE,
}

PRED_INPUT_FN_PARAMS = {
    'tfrecord_dataset_buffer_size': 2,
    'tfrecord_dataset_num_parallel_reads': None,
    'parallel_interleave_cycle_length': 32,
    'parallel_interleave_block_length': 1,
    'parallel_interleave_buffer_output_elements': None,
    'parallel_interleave_prefetch_input_elements': None,
    'map_and_batch_num_parallel_calls': 4,
    'transpose_num_parallel_calls': 4,
    'prefetch_buffer_size': 2,
}

# The mean and stds for each of the channels
GLOBAL_PIXEL_STATS = (np.array([6.74696984, 14.74640167, 10.51260864,
                                10.45369445,  5.49959796, 9.81545561]),
                       np.array([7.95876312, 12.17305868, 5.86172946,
                                 7.83451711, 4.701167, 5.43130431]))


def dummy_pad_files(real, dummy, batch_size):
    to_pad = math.ceil(batch_size / 277)
    return np.concatenate([real.tolist(), dummy[:to_pad]])


def parse_identifier(label):
    return label.split(":")

def main(use_tpu,
         tpu,
         gcp_project,
         tpu_zone,
         url_base_path,
         use_cache,
         model_dir,
         train_epochs,
         train_batch_size,
         num_train_images,
         epochs_per_loop,
         log_step_count_epochs,
         num_cores,
         data_format,
         transpose_input,
         tf_precision,
         n_classes,
         momentum,
         weight_decay,
         base_learning_rate,
         warmup_epochs,
         input_fn_params=DEFAULT_INPUT_FN_PARAMS,
         train_df=None,
         test_df=None,
         valid_pct=.2,
         model='resnet',
         model_depth=50,
         valid_steps=16,
         pred_batch_size=64,
         dim=512,
         pred_on_tpu=False):
    if use_tpu & (tpu is None):
        tpu = os.getenv('TPU_NAME')
    tf.logging.info('tpu: {}'.format(tpu))
    if gcp_project is None:
        gcp_project = os.getenv('TPU_PROJECT')
    tf.logging.info('gcp_project: {}'.format(gcp_project))

    steps_per_epoch = (num_train_images // train_batch_size)
    train_steps = steps_per_epoch * train_epochs
    current_step = estimator._load_global_step_from_checkpoint_dir(
        model_dir)  # pylint: disable=protected-access,line-too-long
    iterations_per_loop = steps_per_epoch * epochs_per_loop
    log_step_count_steps = steps_per_epoch * log_step_count_epochs

    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        tpu if (tpu or use_tpu) else '', zone=tpu_zone, project=gcp_project)

    config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=model_dir,
        save_summary_steps=iterations_per_loop,
        save_checkpoints_steps=iterations_per_loop,
        log_step_count_steps=log_step_count_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=num_cores,
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.
                PER_HOST_V2))  # pylint: disable=line-too-long

    momentum_optimizer = tf.train.MomentumOptimizer(learning_rate=base_learning_rate,
                                           momentum=momentum,
                                           use_nesterov=True)

    adam_optimizer = tf.train.AdamOptimizer(base_learning_rate)

    train_model_fn = functools.partial(
        model_fn,
        n_classes=n_classes,
        num_train_images=num_train_images,
        data_format=data_format,
        transpose_input=transpose_input,
        train_batch_size=train_batch_size,
        iterations_per_loop=iterations_per_loop,
        tf_precision=tf_precision,
        weight_decay=weight_decay,
        base_learning_rate=base_learning_rate,
        warmup_epochs=warmup_epochs,
        model_dir=model_dir,
        use_tpu=use_tpu,
        model_depth=model_depth,
        optimizer=adam_optimizer,
        model=model,
        pred_on_tpu=pred_on_tpu)



    classifier = tf.contrib.tpu.TPUEstimator(
        use_tpu=use_tpu,
        model_fn=train_model_fn,
        config=config,
        train_batch_size=train_batch_size,
        eval_batch_size=train_batch_size,
        predict_batch_size=train_batch_size,
        eval_on_tpu=True,
        export_to_cpu=True)

    use_bfloat16 = (tf_precision == 'bfloat16')

    tfrecord_glob = os.path.join(url_base_path, '*.tfrecord')

    tf.logging.info("Train glob: {}".format(tfrecord_glob))

    train_files, valid_files = rxinput.get_tfrecord_names(url_base_path, train_df, True, valid_pct=valid_pct)

    train_input_fn = functools.partial(rxinput.input_fn,
                                       train_files,
                                       input_fn_params=input_fn_params,
                                       pixel_stats=GLOBAL_PIXEL_STATS,
                                       transpose_input=transpose_input,
                                       use_bfloat16=use_bfloat16,
                                       dim=dim)

    valid_input_fn = functools.partial(rxinput.input_fn,
                                       valid_files,
                                       input_fn_params=input_fn_params,
                                       pixel_stats=GLOBAL_PIXEL_STATS,
                                       transpose_input=transpose_input,
                                       use_bfloat16=use_bfloat16,
                                       dim=dim)

    tf.logging.info('Training for %d steps (%.2f epochs in total). Current'
                    ' step %d.', train_steps, train_steps / steps_per_epoch,
                    current_step)

    start_timestamp = time.time()  # This time will include compilation time

    classifier.train(input_fn=train_input_fn, max_steps=train_steps)

    classifier.evaluate(input_fn=valid_input_fn, steps=valid_steps)

    tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                    train_steps, int(time.time() - start_timestamp))

    elapsed_time = int(time.time() - start_timestamp)
    tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                    train_steps, elapsed_time)

    tf.logging.info('Exporting SavedModel.')

    def serving_input_receiver_fn():
        features = {
            'feature': tf.placeholder(dtype=tf.float32, shape=[None, dim, dim, 6]),
        }
        receiver_tensors = features
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

        classifier.export_saved_model(os.path.join(model_dir, 'saved_model'), serving_input_receiver_fn)

    test_files = rxinput.get_tfrecord_names(url_base_path, test_df)
    all_files = rxinput.get_tfrecord_names(url_base_path, train_df)

    classifier_pred = tf.contrib.tpu.TPUEstimator(
        use_tpu=pred_on_tpu,
        model_fn=train_model_fn,
        config=config,
        train_batch_size=train_batch_size,
        eval_batch_size=train_batch_size,
        predict_batch_size=pred_batch_size,
        eval_on_tpu=pred_on_tpu,
        export_to_cpu=True)

    """
    Kind of hacky, but append on some junk files so we use `drop_remainder` in the dataset to get fixed batch sizes for TPU,
    then we can just ignore any beyond the real amount.
    
    Not sure if I have something configured wrong or what, but TPU prediction is like 100x faster than CPU prediction,
    so I guess a bit of hackiness is worth it?
    """

    pred_params = {'batch_size': pred_batch_size}

    if pred_on_tpu:
        test_files = dummy_pad_files(test_files, all_files, pred_batch_size)

    test_input_fn = functools.partial(rxinput.input_fn,
                                       test_files,
                                       input_fn_params=input_fn_params,
                                       pixel_stats=GLOBAL_PIXEL_STATS,
                                       transpose_input=False,
                                       use_bfloat16=use_bfloat16,
                                       test=True,
                                       dim=dim)

    if pred_on_tpu:
        # Also predict for all the training ones too (with garbage added on the end too) so that we can use that in other models
        all_files = dummy_pad_files(all_files, test_files, pred_batch_size)

    all_input_fn = functools.partial(rxinput.input_fn,
                                       all_files,
                                       input_fn_params=input_fn_params,
                                       pixel_stats=GLOBAL_PIXEL_STATS,
                                       transpose_input=False,
                                       use_bfloat16=use_bfloat16,
                                       test=True,
                                       dim=dim)

    return {
        'test_dataset': test_input_fn(input_fn_params=PRED_INPUT_FN_PARAMS, params=pred_params, id_for_label=True, dim=3).make_one_shot_iterator(),
        'test_predict': classifier_pred.predict(input_fn=test_input_fn, yield_single_examples=False),
        'all_dataset': all_input_fn(input_fn_params=PRED_INPUT_FN_PARAMS, params=pred_params, id_for_label=True, dim=3).make_one_shot_iterator(),
        'all_predict': classifier_pred.predict(input_fn=all_input_fn, yield_single_examples=False)
    }


if __name__ == '__main__':

    p = argparse.ArgumentParser(description='Train ResNet on rxrx1')
    # TPU Parameters
    p.add_argument(
        '--use-tpu',
        type=bool,
        default=True,
        help=('Use TPU to execute the model for training and evaluation. If'
              ' --use_tpu=false, will use whatever devices are available to'
              ' TensorFlow by default (e.g. CPU and GPU)'))
    p.add_argument(
        '--tpu',
        type=str,
        default=None,
        help=(
            'The Cloud TPU to use for training.'
            ' This should be either the name used when creating the Cloud TPU, '
            'or a grpc://ip.address.of.tpu:8470 url.'))
    p.add_argument(
        '--gcp-project',
        type=str,
        default=None,
        help=('Project name for the Cloud TPU-enabled project. '
              'If not specified, we will attempt to automatically '
              'detect the GCE project from metadata.'))
    p.add_argument(
        '--tpu-zone',
        type=str,
        default=None,
        help=('GCE zone where the Cloud TPU is located in. '
              'If not specified, we will attempt to automatically '
              'detect the GCE project from metadata.'))
    p.add_argument('--use-cache', type=bool, default=None)
    # Dataset Parameters
    p.add_argument(
        '--url-base-path',
        type=str,
        default='gs://rxrx1-us-central1/tfrecords/random-42',
        help=('Base path for tfrecord storage bucket url.'))
    # Training parameters
    p.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help=(
            'The Google Cloud Storage bucket where the model and training summaries are'
            ' stored.'))
    p.add_argument(
        '--train-epochs',
        type=int,
        default=1,
        help=(
            'Defining an epoch as one pass through every training example, '
            'the number of total passes through all examples during training. '
            'Implicitly sets the total train steps.'))
    p.add_argument(
        '--num-train-images',
        type=int,
        default=73000
    )
    p.add_argument(
        '--train-batch-size',
        type=int,
        default=512,
        help=('Batch size to use during training.'))
    p.add_argument(
        '--n-classes',
        type=int,
        default=1108,
        help=('The number of label classes - typically will be 1108 '
              'since there are 1108 experimental siRNA classes.'))
    p.add_argument(
        '--epochs-per-loop',
        type=int,
        default=1,
        help=('The number of steps to run on TPU before outfeeding metrics '
              'to the CPU. Larger values will speed up training.'))
    p.add_argument(
        '--log-step-count-epochs',
        type=int,
        default=64,
        help=('The number of epochs at '
              'which global step information is logged .'))
    p.add_argument(
        '--num-cores',
        type=int,
        default=8,
        help=('Number of TPU cores. For a single TPU device, this is 8 because '
              'each TPU has 4 chips each with 2 cores.'))
    p.add_argument(
        '--data-format',
        type=str,
        default='channels_last',
        choices=[
            'channels_first',
            'channels_last',
        ],
        help=('A flag to override the data format used in the model. '
              'To run on CPU or TPU, channels_last should be used. '
              'For GPU, channels_first will improve performance.'))
    p.add_argument(
        '--transpose-input',
        type=bool,
        default=True,
        help=('Use TPU double transpose optimization.'))
    p.add_argument(
        '--tf-precision',
        type=str,
        default='bfloat16',
        choices=['bfloat16', 'float32'],
        help=('Tensorflow precision type used when defining the network.'))

    # Optimizer Parameters

    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument(
        '--base-learning-rate',
        type=float,
        default=0.2,
        help=('Base learning rate when train batch size is 512. '
              'Chosen to match the resnet paper.'))
    p.add_argument(
        '--warmup-epochs',
        type=int,
        default=5,
    )
    args = p.parse_args()
    args = vars(args)
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Parsed args: ')
    for k, v in args.items():
        tf.logging.info('{} : {}'.format(k, v))
    main(**args)
