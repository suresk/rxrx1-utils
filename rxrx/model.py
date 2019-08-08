import tensorflow as tf
from rxrx.official_resnet import resnet_v1
from tensorflow.contrib import summary

def resnet_model_fn(features, labels, mode, params, n_classes, num_train_images,
                    data_format, transpose_input, train_batch_size,
                    momentum, weight_decay, base_learning_rate,  warmup_epochs,
                    use_tpu, iterations_per_loop, model_dir, tf_precision,
                    resnet_depth):
    """The model_fn for ResNet to be used with TPUEstimator.

    Args:
    features: `Tensor` of batched images
    labels: `Tensor` of labels for the data samples
    mode: one of `tf.estimator.ModeKeys.{TRAIN,EVAL,PREDICT}`
    params: `dict` of parameters passed to the model from the TPUEstimator,
        `params['batch_size']` is always provided and should be used as the
        effective batch size.


    Returns:
        A `TPUEstimatorSpec` for the model
    """
    if isinstance(features, dict):
        features = features['feature']

    # In most cases, the default data format NCHW instead of NHWC should be
    # used for a significant performance boost on GPU/TPU. NHWC should be used
    # only if the network needs to be run on CPU since the pooling operations
    # are only supported on NHWC.
    if data_format == 'channels_first':
        assert not transpose_input  # channels_first only for GPU
        features = tf.transpose(features, [0, 3, 1, 2])

    if transpose_input and mode != tf.estimator.ModeKeys.PREDICT:
        features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC

    # This nested function allows us to avoid duplicating the logic which
    # builds the network, for different values of --precision.
    def build_network():
        network = resnet_v1(
            resnet_depth=resnet_depth,
            num_classes=n_classes,
            data_format=data_format)
        return network(
            inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

    if tf_precision == 'bfloat16':
        with tf.contrib.tpu.bfloat16_scope():
            logits = build_network()
        logits = tf.cast(logits, tf.float32)
    elif tf_precision == 'float32':
        logits = build_network()

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            })

    # If necessary, in the model_fn, use params['batch_size'] instead the batch
    # size flags (--train_batch_size or --eval_batch_size).
    batch_size = params['batch_size']  # pylint: disable=unused-variable

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    one_hot_labels = tf.one_hot(labels, n_classes)
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits,
        onehot_labels=one_hot_labels)

    # Add weight decay to the loss for non-batch-normalization variables.
    loss = cross_entropy + weight_decay * tf.add_n([
        tf.nn.l2_loss(v) for v in tf.trainable_variables()
        if 'batch_normalization' not in v.name
    ])

    host_call = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Compute the current epoch and associated learning rate from global_step.
        global_step = tf.train.get_global_step()
        steps_per_epoch = tf.cast(num_train_images / train_batch_size, tf.float32)
        current_epoch = (tf.cast(global_step, tf.float32) / steps_per_epoch)
        warmup_steps = warmup_epochs * steps_per_epoch


        period = 10 * steps_per_epoch
        learning_rate = tf.train.cosine_decay_restarts(base_learning_rate,
                                                       global_step,
                                                       period,
                                                       t_mul=1.0,
                                                       m_mul=1.0,
                                                       alpha=0.0,
                                                       name=None)



        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                               momentum=momentum,
                                               use_nesterov=True)

        if use_tpu:
            # When using TPU, wrap the optimizer with CrossShardOptimizer which
            # handles synchronization details between different TPU cores. To the
            # user, this should look like regular synchronous training.
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        # Batch normalization requires UPDATE_OPS to be added as a dependency to
        # the train operation.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)


        def host_call_fn(gs, loss, lr, ce):
            """Training host call. Creates scalar summaries for training metrics.
            This function is executed on the CPU and should not directly reference
            any Tensors in the rest of the `model_fn`. To pass Tensors from the
            model to the `metric_fn`, provide as part of the `host_call`. See
            https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
            for more information.
            Arguments should match the list of `Tensor` objects passed as the second
            element in the tuple passed to `host_call`.
            Args:
            gs: `Tensor with shape `[batch]` for the global_step
            loss: `Tensor` with shape `[batch]` for the training loss.
            lr: `Tensor` with shape `[batch]` for the learning_rate.
            ce: `Tensor` with shape `[batch]` for the current_epoch.
            Returns:
            List of summary ops to run on the CPU host.
            """
            gs = gs[0]
                # Host call fns are executed FLAGS.iterations_per_loop times after one
                # TPU loop is finished, setting max_queue value to the same as number of
                # iterations will make the summary writer only flush the data to storage
                # once per loop.
            with summary.create_file_writer(model_dir,
                                            max_queue=iterations_per_loop).as_default():
                with summary.always_record_summaries():
                    summary.scalar('loss', loss[0], step=gs)
                    summary.scalar('learning_rate', lr[0], step=gs)
                    summary.scalar('current_epoch', ce[0], step=gs)
                    return summary.all_summary_ops()

            # To log the loss, current learning rate, and epoch for Tensorboard, the
            # summary op needs to be run on the host CPU via host_call. host_call
            # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
            # dimension. These Tensors are implicitly concatenated to
            # [params['batch_size']].
        gs_t = tf.reshape(global_step, [1])
        loss_t = tf.reshape(loss, [1])
        lr_t = tf.reshape(learning_rate, [1])
        ce_t = tf.reshape(current_epoch, [1])

        host_call = (host_call_fn, [gs_t, loss_t, lr_t, ce_t])

    else:
        train_op = None

    eval_metrics = None
    if mode == tf.estimator.ModeKeys.EVAL:

        def metric_fn(labels, logits):
            """Evaluation metric function. Evaluates accuracy.
      This function is executed on the CPU and should not directly reference
      any Tensors in the rest of the `model_fn`. To pass Tensors from the model
      to the `metric_fn`, provide as part of the `eval_metrics`. See
      https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
      for more information.
      Arguments should match the list of `Tensor` objects passed as the second
      element in the tuple passed to `eval_metrics`.
      Args:
        labels: `Tensor` with shape `[batch]`.
        logits: `Tensor` with shape `[batch, num_classes]`.
      Returns:
        A dict of the metrics to return from evaluation.
      """
            predictions = tf.argmax(logits, axis=1)
            top_1_accuracy = tf.metrics.accuracy(labels, predictions)
            in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
            top_5_accuracy = tf.metrics.mean(in_top_5)

            return {
                'top_1_accuracy': top_1_accuracy,
                'top_5_accuracy': top_5_accuracy,
            }

        eval_metrics = (metric_fn, [labels, logits])

    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        host_call=host_call,
        eval_metrics=eval_metrics)
