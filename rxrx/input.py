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
"""Efficient input pipeline using tf.data.Dataset.

Original file:
    https://github.com/tensorflow/tpu/blob/master/models/official/resnet/imagenet_input.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf

DUMMY_SIRNA = 10000


def set_shapes(transpose_input, batch_size, images, labels):
    """Statically set the batch_size dimension."""
    if transpose_input:
        images.set_shape(images.get_shape().merge_with(
            tf.TensorShape([None, None, None, batch_size])))
        labels.set_shape(
            labels.get_shape().merge_with(tf.TensorShape([batch_size])))
    else:
        images.set_shape(images.get_shape().merge_with(
            tf.TensorShape([batch_size, None, None, None])))
        labels.set_shape(
            labels.get_shape().merge_with(tf.TensorShape([batch_size])))

    return images, labels


def record_to_filename(record, url_base, site=1):
    exp = record[0]
    plate = record[1]
    return f'{url_base}/{exp}_p{plate}_s{site}.tfrecord'


def get_tfrecord_names(url_base, df, split=False, valid_pct=0.2, sites=[1,2], seed=None):
    grouped = df.groupby(['experiment', 'plate'])

    train_files = []
    valid_files = []

    for site in sites:
        if split:
            x = grouped.agg({'sirna': 'min'}).reset_index()
            files = x.apply(lambda row: record_to_filename(row, url_base=url_base, site=site), axis=1)
            labels = x['sirna']
            train, valid = train_test_split(files.values, test_size=valid_pct, random_state=seed, stratify=labels)
            train_files = np.concatenate([train_files, train])
            valid_files = np.concatenate([valid_files, valid])
        else:
            train = [record_to_filename(key, url_base=url_base, site=site) for key in grouped.groups.keys()]
            train_files = np.concatenate([train_files, train])

    if split:
        return train_files, valid_files
    else:
        return train_files


def parse(value, test=False, id_for_label=False):

    keys_to_features = {
        'image': tf.FixedLenFeature((), tf.string),
        'well': tf.FixedLenFeature((), tf.string),
        'well_type': tf.FixedLenFeature((), tf.string),
        'plate': tf.FixedLenFeature((), tf.int64),
        'site': tf.FixedLenFeature((), tf.int64),
        'cell_type': tf.FixedLenFeature((), tf.string),
        'experiment': tf.FixedLenFeature((), tf.string)
    }

    if not test:
        keys_to_features['sirna'] = tf.FixedLenFeature((), tf.int64)

    res = tf.parse_single_example(value, keys_to_features)

    if test:
        if id_for_label:
            res['sirna'] = tf.strings.format('{}_{}_{}:{}', (res['experiment'], res['plate'], res['well'], res["site"]))
        else:
            res['sirna'] = DUMMY_SIRNA

    return res


def data_to_image(value, use_bfloat16=True, pixel_stats=None, dim=512):

    image_raw = tf.decode_raw(value['image'], tf.uint8)
    raw_shape = [512, 512, 6]
    image_shape = [dim, dim, 6]
    image = tf.reshape(image_raw, raw_shape)

    if dim != 512:
        image = tf.image.resize(image, [dim, dim])

    image.set_shape(image_shape)

    if pixel_stats is not None:
        mean, std = pixel_stats
        image = (tf.cast(image, tf.float32) - mean) / std

    if use_bfloat16:
        image = tf.image.convert_image_dtype(image, dtype=tf.bfloat16)

    label = value["sirna"]
    return image, label


DEFAULT_PARAMS = dict(batch_size=512)

def input_fn(tf_records_glob,
             input_fn_params,
             params=None,
             use_bfloat16=False,
             pixel_stats = None,
             transpose_input=True,
             shuffle_buffer=64,
             test=False,
             dim=512,
             id_for_label=False):

    batch_size = params['batch_size']

    filenames_dataset = tf.data.Dataset.list_files(tf_records_glob, shuffle=not test)

    def fetch_images(filenames):
        dataset = tf.data.TFRecordDataset(
            filenames,
            compression_type="GZIP",
            buffer_size=(1000 * 1000 *
                         input_fn_params['tfrecord_dataset_buffer_size']),
            num_parallel_reads=input_fn_params[
                'tfrecord_dataset_num_parallel_reads'])
        return dataset

    images_dataset = filenames_dataset.apply(
        tf.contrib.data.parallel_interleave(
            fetch_images,
            cycle_length=input_fn_params['parallel_interleave_cycle_length'],
            block_length=input_fn_params['parallel_interleave_block_length'],
            sloppy=not test,
            buffer_output_elements=input_fn_params[
                'parallel_interleave_buffer_output_elements'],
            prefetch_input_elements=input_fn_params[
                'parallel_interleave_prefetch_input_elements']))

    if not test:
        images_dataset = images_dataset.shuffle(2048).repeat()

    # Get image and label now
    dataset = images_dataset.apply(
        tf.contrib.data.map_and_batch(
            lambda value: data_to_image(parse(value, test, id_for_label=id_for_label), use_bfloat16=use_bfloat16, pixel_stats=pixel_stats, dim=dim),
            batch_size=batch_size,
            num_parallel_calls=input_fn_params['map_and_batch_num_parallel_calls'],
            drop_remainder=True))


    # Transpose for performance on TPU
    if transpose_input:
        dataset = dataset.map(
            lambda images, labels: (tf.transpose(images, [1, 2, 3, 0]), labels),
            num_parallel_calls=input_fn_params['transpose_num_parallel_calls'])

    # Assign static batch size dimension
    dataset = dataset.map(partial(set_shapes, transpose_input, batch_size))

    # Prefetch overlaps in-feed with training
    dataset = dataset.prefetch(
        buffer_size=input_fn_params['prefetch_buffer_size'])

    return dataset
