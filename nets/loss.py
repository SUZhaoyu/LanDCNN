from __future__ import division
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

# def dice_loss(softmax_logits, return_score_maps):
#     eps = tf.constant(1e-6, dtype=tf.float32)
#     # input_score_maps = tf.nn.sigmoid(logits)
#     intersection = tf.reduce_sum(softmax_logits * return_score_maps) + eps
#     union = tf.reduce_sum(softmax_logits) + tf.reduce_sum(return_score_maps) + eps
#     loss = 1. - (2 * intersection / union)
#     tf.summary.scalar('return_dice_loss', loss)
#
#     return loss

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


# def weighted_categorical_crossentropy(output, target, from_logits=False, reuse=None):
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
#         label_id_true = tf.cast(tf.argmax(target, axis=-1), dtype=tf.float32)
#         label_id_pred = tf.cast(tf.argmax(output, axis=-1), dtype=tf.float32)
#
#         weight_mask = tf.cast(tf.not_equal(label_id_pred + label_id_true, tf.constant(FLAGS.background, dtype=tf.float32)), tf.float32)
#         weight_mask *= (FLAGS.weight - 1.)
#         weight_mask += tf.ones_like(pixel_wise_entropy, dtype=tf.float32)
#         weighted_pixel_wise_entropy = pixel_wise_entropy * weight_mask
#         if reuse is None:
#             tf.summary.image('weight_mask', tf.expand_dims(weight_mask, axis=-1))
#
#         return tf.reduce_mean(tf.scalar_mul(tf.constant(10.), weighted_pixel_wise_entropy))
#         # return tf.reduce_mean(weighted_pixel_wise_entropy)
#
#     else:
#         return tf.nn.softmax_cross_entropy_with_logits(labels=target,
#                                                        logits=output)

def categorical_focal_loss(y_true, y_pred, alpha, gamma):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        alpha = np.array(alpha, dtype=np.float32)
        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = tf.constant(1e-6)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * tf.log(y_pred)
        tf.summary.scalar('cross_entropy_sum', tf.reduce_sum(cross_entropy))

        # Calculate Focal Loss
        loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy
        tf.summary.scalar('focal_loss_sum', tf.reduce_sum(loss))

        return loss

def categorical_ce_loss(labels, y_pred):
    y_true = tf.cast(tf.nn.relu(labels), dtype=tf.int32)
    tf.summary.histogram('labels', labels)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return loss


def model_loss(y_pred, labels):
    # loss = categorical_focal_loss(y_true, y_pred, alpha=[1., 10., 10.], gamma=2.)
    loss = categorical_ce_loss(labels, y_pred)
    valid_masks = tf.cast(tf.greater_equal(labels, 0), dtype=tf.float32)
    # weight_masks = tf.cast(tf.greater(labels, 0), dtype=tf.float32) * 9.
    loss = tf.reduce_sum(loss * (valid_masks)) / (tf.reduce_sum(valid_masks) + 1e-6)

    return loss



def dice_coef(y_true, y_pred, smooth = 1e-4):
    y_true_f = tf.layers.flatten(y_true)
    y_pred_f = tf.layers.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f*y_pred_f)
    numerator = 2. * intersection + smooth
    # some implementations don't square y_pred
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    return (numerator/denominator)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)


def categorical_dice_loss(y_true, y_pred, target_cls=[1, 2]):
    total_loss = 0.
    for i in target_cls:
        y_true_cls = tf.cast(tf.equal(y_true, i), dtype=tf.float32)
        y_pred_cls = y_pred[..., i]
        intersection = tf.reduce_sum(y_true_cls * y_pred_cls)
        dice_loss = 1. - (2 * intersection + 1.) / (tf.reduce_sum(y_true_cls) + tf.reduce_sum(y_pred_cls) + 1.)
        total_loss += dice_loss
        tf.summary.scalar('dice_loss_{}'.format(i), dice_loss)
        tf.summary.scalar('dice_loss_true_{}'.format(i), tf.reduce_sum(y_true_cls))
        tf.summary.scalar('dice_loss_pred_{}'.format(i), tf.reduce_sum(y_pred_cls))
        tf.summary.scalar('dice_loss_intersection_{}'.format(i), intersection)

    return total_loss


def dice_loss(y_true, y_pred):
    total_loss = 0.
    y_true = tf.cast(tf.equal(y_true, 1), dtype=tf.float32)
    y_pred = y_pred[..., 0]
    intersection = tf.reduce_sum(y_true * y_pred)
    dice_loss = 1. - (2 * intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1.)
    total_loss += dice_loss
    tf.summary.scalar('dice_loss', dice_loss)
    tf.summary.scalar('dice_true', tf.reduce_sum(y_true))
    tf.summary.scalar('dice_pred', tf.reduce_sum(y_pred))
    tf.summary.scalar('dice_intersection', intersection)

    return total_loss