# -*- coding: utf-8 -*-
from __future__ import division
import time
import os
import numpy as np
import tensorflow as tf
from os.path import join
from tqdm import tqdm

use_DTM = True

tf.app.flags.DEFINE_string('gpu_list', '0,1', '')
tf.app.flags.DEFINE_string('HOME', '/media/data1/LanDCNN','')
tf.app.flags.DEFINE_string('dataset_name', 'preprocessed','')

tf.app.flags.DEFINE_string('task_name', 'building-valid', 'Provided loss funtions: dice, weighted_loss, entropy, weighted_entropy')

tf.app.flags.DEFINE_string('phase', 'validation', '')
tf.app.flags.DEFINE_boolean('DTM', use_DTM, '')
if use_DTM:
    tf.app.flags.DEFINE_integer('num_of_channels', 7, '')
else:
    tf.app.flags.DEFINE_integer('num_of_channels', 6, '')
tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_boolean('continue_train', False, '')
tf.app.flags.DEFINE_float('learning_rate', 0.002, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 15, '')
tf.app.flags.DEFINE_integer('num_of_labels', 2, '')

tf.app.flags.DEFINE_integer('background', 0, '')
tf.app.flags.DEFINE_integer('landslide', 1, '')

tf.app.flags.DEFINE_float('weight', 10., '')
tf.app.flags.DEFINE_string('loss', 'weighted_entropy', '')
tf.app.flags.DEFINE_float('dice_thres', 0.8, '')
tf.app.flags.DEFINE_integer('initial_step', 0, '')

from nets import model
from nets import loss
from utils import read_info_from_txt, dir_concat
import training_generator

FLAGS = tf.app.flags.FLAGS

gpus = [int(gpu) for gpu in FLAGS.gpu_list.split(',')]
print('INFO: Using GPU: {}'.format(gpus))
checkpoint_path = dir_concat(FLAGS.HOME, ['checkpoint', FLAGS.task_name])
mean = np.load(dir_concat(FLAGS.HOME, ['dataset', FLAGS.dataset_name, '{}_mean.npy'.format(FLAGS.phase)]))
std = np.load(dir_concat(FLAGS.HOME, ['dataset', FLAGS.dataset_name, '{}_std.npy'.format(FLAGS.phase)]))
epoch_iters = 1300 // (FLAGS.batch_size_per_gpu * len(gpus))


def scar_accuracy(target_label_maps, softmax_logits):
    epsilon = tf.convert_to_tensor(1, tf.int64)
    if FLAGS.loss == 'dice':
        class_id_true = tf.cast(target_label_maps, dtype=tf.int64)
        class_id_pred = tf.cast(tf.greater(softmax_logits, FLAGS.dice_thres), dtype=tf.int64)
        landslide_mask_true = tf.cast(tf.equal(class_id_true, tf.constant(1, dtype=tf.int64)), tf.int64)
        landslide_mask_pred = tf.cast(tf.equal(class_id_pred, tf.constant(1, dtype=tf.int64)), tf.int64)
    else:
        class_id_true = tf.argmax(target_label_maps, axis=-1)
        class_id_pred = tf.argmax(softmax_logits, axis=-1)
        # Replace class_id_preds with class_id_true for recall here
        landslide_mask_true = tf.cast(tf.equal(class_id_true, tf.constant(FLAGS.landslide, dtype=tf.int64)), tf.int64)
        landslide_mask_pred = tf.cast(tf.equal(class_id_pred, tf.constant(FLAGS.landslide, dtype=tf.int64)), tf.int64)
    correct_pred_tensor = tf.cast(tf.equal(landslide_mask_true, landslide_mask_pred), tf.int64) * landslide_mask_true
    wrong_pred_tensor = tf.cast(tf.not_equal(landslide_mask_true, landslide_mask_pred), tf.int64) * tf.cast(tf.not_equal(
        landslide_mask_true + landslide_mask_pred, tf.constant(FLAGS.background, dtype=tf.int64)), tf.int64)
    class_acc = (tf.reduce_sum(correct_pred_tensor) + epsilon) / (tf.reduce_sum(correct_pred_tensor) + tf.reduce_sum(wrong_pred_tensor) + epsilon)
    return class_acc, landslide_mask_pred

def tower_loss(input_images, target_label_maps, weight_maps, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        softmax_logits = model.model(input_images, is_training=True)
    if FLAGS.loss == 'dice':
        model_loss = loss.weighted_dice_loss(softmax_logits, target_label_maps, reuse=reuse_variables)
    elif FLAGS.loss == 'entropy':
        model_loss = loss.categorical_crossentropy(softmax_logits, target_label_maps, from_logits=False)
    else:
        model_loss = loss.weighted_categorical_crossentropy(softmax_logits, target_label_maps, weight_maps, from_logits=False, reuse=reuse_variables)
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    landslide_acc, landslide_map = scar_accuracy(target_label_maps, softmax_logits)

    if reuse_variables is None:
        tf.summary.image('prior image', (input_images[:, :, :, :3] * std[:3]) + mean[:3])
        tf.summary.image('posterior image', (input_images[:, :, :, 3:6] * std[3:6]) + mean[3:6])
        if FLAGS.DTM:
            tf.summary.image('DTM', tf.expand_dims((input_images[:, :, :, -1] * std[6]) + mean[6], axis=-1))

        if FLAGS.loss =='dice':
            tf.summary.image('return label map', tf.expand_dims(softmax_logits[:, :, :, 0], axis=-1) * 255)
            tf.summary.image('target label map', tf.expand_dims(target_label_maps[:, :, :, 0], axis=-1) * 255)
            tf.summary.image('return landslide label', tf.cast(landslide_map, dtype=tf.float32) * 255)
        else:
            tf.summary.image('return label map 0', tf.expand_dims(softmax_logits[:, :, :, 0], axis=-1) * 255)
            tf.summary.image('return label map 1', tf.expand_dims(softmax_logits[:, :, :, 1], axis=-1) * 255)
            tf.summary.image('target label map 0', tf.expand_dims(target_label_maps[:, :, :, 0], axis=-1) * 255)
            tf.summary.image('target label map 1', tf.expand_dims(target_label_maps[:, :, :, 1], axis=-1) * 255)
            tf.summary.image('return landslide label', tf.expand_dims(tf.cast(landslide_map, dtype=tf.float32), axis=-1) * 255)

        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('landslide_accuracy', landslide_acc)

    return total_loss, model_loss, landslide_acc


def average_grads(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):

        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads




def main(argv=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_list
    if tf.gfile.Exists(checkpoint_path) and not FLAGS.continue_train:
        tf.gfile.DeleteRecursively(checkpoint_path)
        tf.gfile.MkDir(checkpoint_path)
    else:
        tf.gfile.MkDir(checkpoint_path)

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, FLAGS.num_of_channels], name='input_images')
    weight_maps = tf.placeholder(tf.float32, shape=[None, None, None], name='weight_maps')
    if FLAGS.loss == 'dice':
        input_label_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_label_maps')
    else:
        input_label_maps = tf.placeholder(tf.float32, shape=[None, None, None, FLAGS.num_of_labels], name='input_label_maps')

    input_images_split = tf.split(input_images, len(gpus))
    input_label_maps_split = tf.split(input_label_maps, len(gpus))
    weight_maps_split = tf.split(weight_maps, len(gpus))
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(FLAGS.initial_step), trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=500, decay_rate=0.5, staircase=True)
    tf.summary.scalar('learning rate', learning_rate)

    optimizer = tf.train.AdamOptimizer(learning_rate)

    tower_grads = []
    reuse_variables = None

    for i, gpu_id in enumerate(gpus):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('model_%d' % gpu_id) as scope:

                total_loss, model_loss, landslide_acc = tower_loss(input_images_split[i], input_label_maps_split[i], weight_maps_split[i], reuse_variables)
                batch_norm_update_operation = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                reuse_variables = True

                grads = optimizer.compute_gradients(total_loss)
                tower_grads.append(grads)

    grads = average_grads(tower_grads)
    apply_grads_operation = optimizer.apply_gradients(grads, global_step=global_step)
    summary_operation = tf.summary.merge_all()

    variables_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
    apply_variables_averages_operation = variables_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_variables_averages_operation, apply_grads_operation, batch_norm_update_operation]):
        train_operation = tf.no_op(name='train_operation')

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(checkpoint_path, tf.get_default_graph())

    init = tf.global_variables_initializer()

    data_generator = training_generator.get_batch(num_workers=6, max_queue_size=12, batch_size=FLAGS.batch_size_per_gpu * len(gpus))


    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True



    with tf.Session(config=config) as sess:
        if FLAGS.continue_train:
            saved_weight = join(checkpoint_path, 'model_latest.ckpt')
            print('INFO: Continue to train with model saved in {}'.format(saved_weight))
            saver.restore(sess, saved_weight)

        else:
            sess.run(init)

        data_load_time, training_time = 0, 0
        epoch, step = 0, 0
        epoch_model_loss, epoch_total_loss, epoch_acc = 0, 0, 0
        while epoch < 500:
            print('epoch: {:03d}'.format(epoch))
            for _ in tqdm(range(epoch_iters)):

                iter_start_time = time.time()
                data = next(data_generator)
                iter_data_load_time = time.time() - iter_start_time
                data_load_time += iter_data_load_time
                ml, tl, acc, lr, _, summary, step = sess.run([model_loss, total_loss, landslide_acc, learning_rate, train_operation, summary_operation, global_step],
                                              feed_dict={input_images: data[0], input_label_maps: data[1], weight_maps: data[2]})
                epoch_model_loss += ml
                epoch_total_loss += tl
                epoch_acc += acc
                iter_training_time = time.time() - iter_start_time - iter_data_load_time
                training_time += iter_training_time
                summary_writer.add_summary(summary, global_step=step)

            # _, ml, tl, acc, lr, summary = sess.run([train_operation, model_loss, total_loss, landslide_acc, learning_rate, summary_operation],
            #                                     feed_dict={input_images: data[0], input_label_maps: data[1]})
            print('epoch {:03d}: model_loss={:.4f}, total_loss={:.4f}, accuracy={:.4f}, learning rate={:.6f}'
                  .format(epoch, epoch_model_loss / epoch_iters, epoch_total_loss / epoch_iters, epoch_acc / epoch_iters, lr))
            print('Data load time per image: {:.2f} ms, Network training time per image: {:.2f} ms'
                  .format((training_time * 1e3 / (FLAGS.batch_size_per_gpu * len(gpus) * epoch_iters)),
                          (data_load_time * 1e3 / (FLAGS.batch_size_per_gpu * len(gpus) * epoch_iters))))
            data_load_time, training_time = 0, 0
            epoch_model_loss, epoch_total_loss, epoch_acc = 0, 0, 0
            summary_writer.add_summary(summary, global_step=epoch)
            saver.save(sess, join(checkpoint_path, 'model_latest.ckpt'))

            if epoch % 10 == 0 and epoch != 0:
                # print('INFO: Start validation on the validation dataset.')
                # os.system()
                saver.save(sess, join(checkpoint_path, 'model_{:03d}.ckpt'.format(epoch)))


            epoch += 1

if __name__ == '__main__':
    tf.app.run()


