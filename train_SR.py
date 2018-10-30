import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import cv2
from dataset import SR_data_load
import SR_models
import losses
import utils

tf.app.flags.DEFINE_string('run_gpu', '0', 'use single gpu')
tf.app.flags.DEFINE_string('save_path', '/where/your/folder', '')
tf.app.flags.DEFINE_boolean('model_restore', False, '')
tf.app.flags.DEFINE_string('image_path', '/where/your/saved/image/folder', '')
tf.app.flags.DEFINE_integer('batch_size', 32, '')
tf.app.flags.DEFINE_integer('num_readers', 4, '')
tf.app.flags.DEFINE_integer('input_size', 32, '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'define your learing strategy')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('vgg_path', None, '/where/your/vgg_19.ckpt')
tf.app.flags.DEFINE_integer('num_workers', 4, '')
tf.app.flags.DEFINE_integer('max_to_keep', 10, 'how many do you want to save models?')
tf.app.flags.DEFINE_integer('save_model_steps', 10000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 10, '')
tf.app.flags.DEFINE_integer('max_steps', 1000000, '')
tf.app.flags.DEFINE_string('losses', 'perceptual', 'mse,perceptual,texture,adv')
tf.app.flags.DEFINE_string('adv_direction', 'g2d', 'g2d or d2g')
tf.app.flags.DEFINE_float('adv_gen_w', 0.001, '')
tf.app.flags.DEFINE_float('adv_disc_w', 1.0, '')
FLAGS = tf.app.flags.FLAGS

def main(argv=None):
    ######################### System setup
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.run_gpu
    utils.prepare_checkpoint_path(FLAGS.save_path, FLAGS.model_restore)
    
    ######################### Model setup
    low_size = FLAGS.input_size
    high_size = int(FLAGS.input_size * FLAGS.SR_scale)

    input_low_images = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, low_size, low_size, 3], name='input_low_images')
    input_high_images = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, high_size, high_size, 3], name='input_high_images')


    model_builder = SR_models.model_builder()

    generated_high_images, resized_low_images = model_builder.generator(input_low_images, is_training=True, model=FLAGS.model)
    
    tf.summary.image('1_input_low_images', input_low_images)
    tf.summary.image('2_input_high_images', input_high_images)
    vis_gen_images = tf.cast(tf.clip_by_value(generated_high_images, 0, 255), tf.uint8)
    tf.summary.image('3_generated_images', vis_gen_images)
    tf.summary.image('4_bicubic_images', resized_low_images)
    vis_high_images = tf.cast(input_high_images, tf.uint8)
    vis_gen_images = tf.concat([resized_low_images, vis_gen_images, vis_high_images], axis=2)
    tf.summary.image('5_bicubic_gen_gt', vis_gen_images)
    
    
    ######################### Losses setup
    loss_builder = losses.loss_builder()

    loss_list = utils.loss_parser(FLAGS.losses)
    generator_loss = 0.0

    if 'mse' in loss_list or 'l2' in loss_list or 'l2_loss' in loss_list:
        mse_loss = loss_builder.get_loss(input_high_images, generated_high_images, type='mse')
        generator_loss = generator_loss +  1.0 * mse_loss
        tf.summary.scalar('mse_loss', mse_loss)
    if 'inverse_mse' in loss_list:
        inv_mse_loss = loss_builder.get_loss(input_low_images, generated_high_images, type='inverse_mse')
        generator_loss = generator_loss + 100.0 * inv_mse_loss
        tf.summary.scalar('inv_mse_loss', inv_mse_loss)
    if 'fft_mse' in loss_list:
        fft_mse_loss = loss_builder.get_loss(input_high_images, generated_high_images, type='fft_mse')
        generator_loss = generator_loss + 1.0 * fft_mse_loss
        tf.summary.scalar('fft_mse_loss', fft_mse_loss)
    if 'l1' in loss_list or 'l1_loss' in loss_list:
        l1_loss = loss_builder.get_loss(input_high_images, generated_high_images, type='l1_loss')
        generator_loss = generator_loss + 0.01 * l1_loss
        tf.summary.scalar('l1_loss', l1_loss)
    if 'perceptual' in loss_list:
        pl_pool5 = loss_builder.get_loss(input_high_images, generated_high_images, type='perceptual')
        pl_pool5 *= 2e-2
        generator_loss = generator_loss + pl_pool5
        tf.summary.scalar('pl_pool5', pl_pool5)
    if 'texture' in loss_list:
        tl_conv1, tl_conv2, tl_conv3 = loss_builder.get_loss(input_high_images, generated_high_images, type='texture')
        
        #generator_loss = generator_loss + 1e-2 * tl_conv1 + 1e-2 * tl_conv2 + 1e-2 * tl_conv3
        tl_weight = 10.0
        #generator_loss = generator_loss + tl_weight * tl_conv1 + tl_weight * tl_conv2 + tl_weight * tl_conv3
        generator_loss = generator_loss + tl_weight * tl_conv3

        tf.summary.scalar('tl_conv1', tl_conv1)
        tf.summary.scalar('tl_conv2', tl_conv2)
        tf.summary.scalar('tl_conv3', tl_conv3)
    if 'adv' in loss_list:
        adv_gen_loss, adv_disc_loss = loss_builder.get_loss(input_high_images, generated_high_images, type='adv')
        tf.summary.scalar('adv_gen', adv_gen_loss)
        tf.summary.scalar('adv_disc', adv_disc_loss)
        discrim_loss = FLAGS.adv_disc_w *  adv_disc_loss
        generator_loss = generator_loss + FLAGS.adv_gen_w * adv_gen_loss

    tf.summary.scalar('generator_loss', generator_loss)
    
    ######################### Training setup
    global_step = tf.get_variable('global_step', [], dtype=tf.int64, initializer=tf.constant_initializer(0), trainable=False)

    train_vars = tf.trainable_variables()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    learning_rate = utils.configure_learning_rate(FLAGS.learning_rate, global_step)
    
    #gen_optimizer = utils.configure_optimizer(learning_rate)
    #gen_gradients = gen_optimizer.compute_gradients(generator_loss, var_list=generator_vars)
    #gen_grad_updates = gen_optimizer.apply_gradients(gen_gradients)#, global_step=global_step)

    if 'adv' in loss_list:
        discrim_vars = [var for var in train_vars if var.name.startswith('discriminator')]
        disc_optimizer = utils.configure_optimizer(learning_rate)
        disc_gradients = disc_optimizer.compute_gradients(discrim_loss, var_list=discrim_vars)
        disc_grad_updates = disc_optimizer.apply_gradients(disc_gradients, global_step=global_step)
        
        with tf.control_dependencies([disc_grad_updates] + update_ops):
            generator_vars = [var for var in train_vars if var.name.startswith('generator')]
            gen_optimizer = utils.configure_optimizer(learning_rate)
            gen_gradients = gen_optimizer.compute_gradients(generator_loss, var_list=generator_vars)
            gen_grad_updates = gen_optimizer.apply_gradients(gen_gradients, global_step=global_step)
            
            train_op = gen_grad_updates

    else:

        generator_vars = [var for var in train_vars if var.name.startswith('generator')]
        discrim_vars = generator_vars
        gen_optimizer = utils.configure_optimizer(learning_rate)
        gen_gradients = gen_optimizer.compute_gradients(generator_loss, var_list=generator_vars)
        gen_grad_updates = gen_optimizer.apply_gradients(gen_gradients)#, global_step=global_step)
        with tf.control_dependencies([gen_grad_updates] + update_ops):
            train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.save_path, tf.get_default_graph())

    ######################### Train process
    data_generator = SR_data_load.get_batch(image_path=FLAGS.image_path,
                                            num_workers=FLAGS.num_workers,
                                            batch_size=FLAGS.batch_size,
                                            hr_size=high_size)

    ## vgg_stop process
    #utils.print_vars(train_vars)
    #utils.print_vars(generator_vars)
    if FLAGS.vgg_path is not None:
        variable_restore_op = utils.get_restore_op(FLAGS.vgg_path, train_vars)
        
    #############
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if FLAGS.model_restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.save_path)
            saver.restore(sess, ckpt)
        else:
            sess.run(tf.global_variables_initializer())
            if FLAGS.vgg_path is not None:
                variable_restore_op(sess)
        
        start_time = time.time()
        for iter_val in range(int(global_step.eval()) + 1, FLAGS.max_steps + 1):
            data = next(data_generator)
            high_images = np.asarray(data[0])
            low_images = np.asarray(data[1])

            feed_dict = {input_low_images: low_images,
                         input_high_images: high_images}

            generator_loss_val, _, g_w, d_w = sess.run([generator_loss, train_op, generator_vars[0], discrim_vars[0]], feed_dict=feed_dict)
            
            if iter_val != 0 and iter_val % FLAGS.save_summary_steps == 0:
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step=iter_val)
                
                used_time = time.time() - start_time
                avg_time_per_step = used_time / FLAGS.save_summary_steps
                avg_examples_per_second = (FLAGS.save_summary_steps * FLAGS.batch_size) / used_time

                print('step %d, generator_loss %.4f, weights %.2f, %.2f, %.2f seconds/step, %.2f examples/second'
                      % (iter_val, generator_loss_val, np.sum(g_w), np.sum(d_w), avg_time_per_step, avg_examples_per_second))
                start_time = time.time()

            if iter_val != 0 and iter_val % FLAGS.save_model_steps == 0:
                checkpoint_fn = os.path.join(FLAGS.save_path, 'model.ckpt')
                saver.save(sess, checkpoint_fn, global_step=iter_val)

        print('')
        print('*' * 30)
        print(' Training done!!! ')
        print('*' * 30)
        print('')

if __name__ == '__main__':
    tf.app.run()
