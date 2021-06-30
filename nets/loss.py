from __future__ import division
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

def dice_loss(softmax_logits, return_score_maps):
    eps = tf.constant(1e-6, dtype=tf.float32)
    # input_score_maps = tf.nn.sigmoid(logits)
    intersection = tf.reduce_sum(softmax_logits * return_score_maps) + eps
    union = tf.reduce_sum(softmax_logits) + tf.reduce_sum(return_score_maps) + eps
    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('return_dice_loss', loss)

    return loss

def weighted_dice_loss(softmax_logits, target, reuse=None):
    label_id_true = target
    label_id_pred = tf.cast(tf.greater(softmax_logits, FLAGS.dice_thres), dtype=tf.float32)

    weight_mask = tf.cast(tf.not_equal(label_id_pred + label_id_true, tf.constant(FLAGS.background, dtype=tf.float32)), tf.float32)
    weight_mask *= (FLAGS.weight - 1.)
    weight_mask += tf.ones_like(softmax_logits, dtype=tf.float32)
    if reuse is None:
        tf.summary.image('weight_mask', weight_mask)
    eps = tf.constant(1e-6, dtype=tf.float32)
    # input_score_maps = tf.nn.sigmoid(logits)
    intersection = tf.reduce_sum(softmax_logits * target * weight_mask) + eps
    union = tf.reduce_sum(softmax_logits * weight_mask) + tf.reduce_sum(target * weight_mask) + eps
    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('return_dice_loss', loss)

    return loss

def categorical_crossentropy(output, target, from_logits=False):
    """Categorical crossentropy between an output tensor and a target tensor.
    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
    # Returns
        Output tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output,
                                len(output.get_shape()) - 1,
                                True)
        # manual computation of crossentropy
        _epsilon = tf.convert_to_tensor(1e-7, output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)

        pixel_wise_entropy = - tf.reduce_sum(target * tf.log(output),
                                             len(output.get_shape()) - 1)
        return tf.reduce_mean(tf.scalar_mul(tf.constant(4.), pixel_wise_entropy))

    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                       logits=output)


# def weighted_categorical_crossentropy(output, target, weight, from_logits=False):
#     # Note: tf.nn.softmax_cross_entropy_with_logits
#     # expects logits, Keras expects probabilities.
#
#     if not from_logits:
#         # scale preds so that the class probas of each sample sum to 1
#         output /= tf.reduce_sum(output,
#                                 len(output.get_shape()) - 1,
#                                 True)
#         # manual computation of crossentropy
#         _epsilon = tf.convert_to_tensor(1e-7, output.dtype.base_dtype)
#         output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
#
#         pixel_wise_entropy = - tf.reduce_sum(target * tf.log(output),
#                                              len(output.get_shape()) - 1)
#         weighted_pixel_wise_entropy = pixel_wise_entropy * weight
#         return tf.reduce_mean(tf.scalar_mul(tf.constant(4.), weighted_pixel_wise_entropy))
#
#     else:
#         return tf.nn.softmax_cross_entropy_with_logits(labels=target,
#                                                        logits=output)


def weighted_categorical_crossentropy(output, target, weights, from_logits=False, reuse=None):
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.

    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output,
                                len(output.get_shape()) - 1,
                                True)
        # manual computation of crossentropy
        _epsilon = tf.convert_to_tensor(1e-7, output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)

        pixel_wise_entropy = - tf.reduce_sum(target * tf.log(output),
                                             len(output.get_shape()) - 1)
        label_id_true = tf.cast(tf.argmax(target, axis=-1), dtype=tf.float32)
        label_id_pred = tf.cast(tf.argmax(output, axis=-1), dtype=tf.float32)

        weight_mask = tf.cast(tf.not_equal(label_id_pred + label_id_true, tf.constant(FLAGS.background, dtype=tf.float32)), tf.float32)
        weight_mask *= (FLAGS.weight - 1.)
        weight_mask += tf.ones_like(pixel_wise_entropy, dtype=tf.float32)
        weighted_pixel_wise_entropy = pixel_wise_entropy * weight_mask * weights
        if reuse is None:
            tf.summary.image('weight_mask', tf.expand_dims(weight_mask, axis=-1))
            tf.summary.image('weight', tf.expand_dims(weights, axis=-1))

        return tf.reduce_mean(tf.scalar_mul(tf.constant(10.), weighted_pixel_wise_entropy))
        # return tf.reduce_mean(weighted_pixel_wise_entropy)

    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                       logits=output)

