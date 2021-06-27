# -*- coding: utf-8 -*-
from __future__ import division
import time
import os
import tensorflow as tf
from os.path import join
from tqdm import tqdm


tf.app.flags.DEFINE_string('gpu_list', '0,1', '')
tf.app.flags.DEFINE_string('HOME', '/media/data1/LanDCNN', '')

tf.app.flags.DEFINE_string('task_name', 'dice', '')

tf.app.flags.DEFINE_integer('num_of_channels', 7, '')
tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_boolean('continue_train', False, '')
tf.app.flags.DEFINE_float('learning_rate', 0.0002, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.95, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 16, '')
tf.app.flags.DEFINE_integer('num_of_labels', 3, '')

tf.app.flags.DEFINE_integer('background_id', 0, '')
tf.app.flags.DEFINE_integer('landslide_id', 1, '')
tf.app.flags.DEFINE_integer('building_id', 2, '')

tf.app.flags.DEFINE_float('weight', 10., '')
tf.app.flags.DEFINE_float('dice_thres', 0.8, '')
tf.app.flags.DEFINE_integer('initial_step', 0, '')

from nets import model
from nets import loss
from utils import dir_concat
from training_generator import Dataset

TrainingDataset = Dataset(task='training', num_worker=1)
data_generator = TrainingDataset.train_generator()

FLAGS = tf.app.flags.FLAGS

gpus = [int(gpu) for gpu in FLAGS.gpu_list.split(',')]
print('INFO: Using GPU: {}'.format(gpus))
checkpoint_path = dir_concat(FLAGS.HOME, ['checkpoint', FLAGS.task_name])
os.system('rm -rf {}'.format(checkpoint_path))
os.mkdir(checkpoint_path)
epoch_iters = 1000 // (FLAGS.batch_size_per_gpu * len(gpus))


def get_iou(target_label_maps, softmax_logits):
    epsilon = tf.convert_to_tensor(1, tf.float32)
    pred_labels = tf.argmax(tf.nn.softmax(softmax_logits), axis=-1)

    landslide_gt = tf.cast(tf.equal(target_label_maps, FLAGS.landslide_id), dtype=tf.float32)
    buildings_gt = tf.cast(tf.equal(target_label_maps, FLAGS.building_id), dtype=tf.float32)
    landslide_pred = tf.cast(tf.equal(pred_labels, FLAGS.landslide_id), dtype=tf.float32)
    buildings_pred = tf.cast(tf.equal(pred_labels, FLAGS.building_id), dtype=tf.float32)

    landslide_correct = landslide_gt * landslide_pred
    buildings_correct = buildings_gt * buildings_pred

    landslide_union = tf.cast(tf.greater(landslide_gt + landslide_pred, 0), dtype=tf.float32)
    buildings_union = tf.cast(tf.greater(buildings_gt + buildings_pred, 0), dtype=tf.float32)

    landslide_iou = tf.reduce_sum(landslide_correct) / (tf.reduce_sum(landslide_union) + epsilon)
    buildings_iou = tf.reduce_sum(buildings_correct) / (tf.reduce_sum(buildings_union) + epsilon)

    tf.summary.scalar('landslide_correct', tf.reduce_sum(landslide_correct))
    tf.summary.scalar('landslide_union', tf.reduce_sum(landslide_union))
    tf.summary.scalar('buildings_correct', tf.reduce_sum(buildings_correct))
    tf.summary.scalar('buildings_union', tf.reduce_sum(buildings_union))

    return landslide_iou, buildings_iou

def tower_loss(input_images, target_label_maps, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        softmax_logits = model.model(input_images, is_training=True)
        # model_loss = loss.model_loss(softmax_logits, target_label_maps)
        model_loss = loss.categorical_dice_loss(y_true=target_label_maps, y_pred=softmax_logits)
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    landslide_iou, buildings_iou = get_iou(target_label_maps, softmax_logits)
    avg_iou = (landslide_iou + buildings_iou) / 2.

    if reuse_variables is None:
        one_hot_label_maps = tf.one_hot(tf.cast(tf.nn.relu(target_label_maps), dtype=tf.int32), depth=3)

        tf.summary.image('return label map 0', tf.expand_dims(softmax_logits[:, :, :, 0], axis=-1) * 255)
        tf.summary.image('return label map 1', tf.expand_dims(softmax_logits[:, :, :, 1], axis=-1) * 255)
        tf.summary.image('return label map 2', tf.expand_dims(softmax_logits[:, :, :, 2], axis=-1) * 255)
        tf.summary.image('target label map 0', tf.expand_dims(one_hot_label_maps[:, :, :, 0], axis=-1) * 255)
        tf.summary.image('target label map 1', tf.expand_dims(one_hot_label_maps[:, :, :, 1], axis=-1) * 255)
        tf.summary.image('target label map 2', tf.expand_dims(one_hot_label_maps[:, :, :, 2], axis=-1) * 255)

        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('landslide_iou', landslide_iou)
        tf.summary.scalar('buildings_iou', buildings_iou)

    return total_loss, model_loss, avg_iou


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
    # if tf.gfile.Exists(checkpoint_path) and not FLAGS.continue_train:
    #     tf.gfile.DeleteRecursively(checkpoint_path)
    #     tf.gfile.MkDir(checkpoint_path)
    # else:
    #     tf.gfile.MkDir(checkpoint_path)

    input_images = tf.placeholder(tf.float32, shape=[None, FLAGS.input_size, FLAGS.input_size, FLAGS.num_of_channels], name='input_images')
    input_label_maps = tf.placeholder(tf.float32, shape=[None, FLAGS.input_size, FLAGS.input_size], name='input_label_maps')

    input_images_split = tf.split(input_images, len(gpus))
    input_label_maps_split = tf.split(input_label_maps, len(gpus))
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(FLAGS.initial_step), trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=600, decay_rate=0.8, staircase=True)
    tf.summary.scalar('learning rate', learning_rate)

    optimizer = tf.train.AdamOptimizer(learning_rate)

    tower_grads = []
    reuse_variables = None

    for i, gpu_id in enumerate(gpus):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('model_%d' % gpu_id) as scope:

                total_loss, model_loss, avg_iou = tower_loss(input_images_split[i], input_label_maps_split[i], reuse_variables)
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

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = False



    with tf.Session(config=config) as sess:
        if FLAGS.continue_train:
            saved_weight = join(checkpoint_path, 'model_latest.ckpt')
            print('INFO: Continue to train with model saved in {}'.format(saved_weight))
            saver.restore(sess, saved_weight)

        else:
            sess.run(init)

        data_load_time, training_time = 0, 0
        epoch, step = 0, 0
        epoch_model_loss, epoch_total_loss, epoch_iou = 0, 0, 0
        while epoch < 500:
            print('epoch: {:03d}'.format(epoch))
            for _ in tqdm(range(epoch_iters)):

                iter_start_time = time.time()
                data = next(data_generator)
                iter_data_load_time = time.time() - iter_start_time
                data_load_time += iter_data_load_time
                ml, tl, iou, lr, _, summary, step = sess.run([model_loss, total_loss, avg_iou, learning_rate, train_operation, summary_operation, global_step],
                                              feed_dict={input_images: data[0],
                                                         input_label_maps: data[1]})
                epoch_model_loss += ml
                epoch_total_loss += tl
                epoch_iou += iou
                iter_training_time = time.time() - iter_start_time - iter_data_load_time
                training_time += iter_training_time
                summary_writer.add_summary(summary, global_step=step)

            # _, ml, tl, iou, lr, summary = sess.run([train_operation, model_loss, total_loss, avg_iou, learning_rate, summary_operation],
            #                                         feed_dict={input_images: data[0],
            #                                                    input_label_maps: data[1]})
            print('epoch {:03d}: model_loss={:.4f}, total_loss={:.4f}, accuracy={:.4f}, learning rate={:.6f}'
                  .format(epoch, epoch_model_loss / epoch_iters, epoch_total_loss / epoch_iters, epoch_iou / epoch_iters, lr))
            print('Data load time per image: {:.2f} ms, Network training time per image: {:.2f} ms'
                  .format((training_time * 1e3 / (FLAGS.batch_size_per_gpu * len(gpus) * epoch_iters)),
                          (data_load_time * 1e3 / (FLAGS.batch_size_per_gpu * len(gpus) * epoch_iters))))
            data_load_time, training_time = 0, 0
            epoch_model_loss, epoch_total_loss, epoch_iou = 0, 0, 0

            saver.save(sess, join(checkpoint_path, 'model_latest.ckpt'))

            if epoch % 10 == 0 and epoch != 0:
                # print('INFO: Start validation on the validation dataset.')
                # os.system()
                saver.save(sess, join(checkpoint_path, 'model_{:03d}.ckpt'.format(epoch)))
            epoch += 1

if __name__ == '__main__':
    tf.app.run()


