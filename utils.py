import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import os
import glob

FLAGS = tf.app.flags.FLAGS

def prepare_checkpoint_path(save_path, restore):
    if not tf.gfile.Exists(save_path):
        tf.gfile.MkDir(save_path)
    else:
        if not restore:
            tf.gfile.DeleteRecursively(save_path)
            tf.gfile.MkDir(save_path)

def configure_learning_rate(learning_rate_init_value, global_step):
    learning_rate = tf.train.exponential_decay(learning_rate_init_value, global_step, decay_steps=10000, decay_rate=0.94, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    return learning_rate

def configure_optimizer(learning_rate):
    return tf.train.AdamOptimizer(learning_rate)

def get_restore_op(vgg_path, train_vars, check=False):
    vgg_19_vars = [var for var in train_vars if var.name.startswith('vgg')]
    if check:
        print_vars(vgg_19_vars)
    variable_restore_op = slim.assign_from_checkpoint_fn(
                            vgg_path,
                            vgg_19_vars,
                            ignore_missing_vars=True)
    return variable_restore_op

def print_vars(var_list):
    print('')
    for var in var_list:
        print(var)
    print('')

def loss_parser(str_loss):
    '''
    NOTE!!! str_loss should be like 'mse,perceptual,texture,adv'...
    '''
    selected_loss_array = str_loss.split(',')
    return selected_loss_array

def get_last_ckpt_path(folder_path):
    '''
    folder_path = .../where/your/saved/model/folder
    '''

    meta_paths = sorted(glob.glob(os.path.join(folder_path, '*.meta')))

    numbers = []

    for meta_path in meta_paths:
        numbers.append(int(meta_path.split('-')[-1].split('.')[0]))

    numbers = np.asarray(numbers)

    sorted_idx = np.argsort(numbers)

    latest_meta_path = meta_paths[sorted_idx[-1]]

    ckpt_path = latest_meta_path.replace('.meta', '')

    return ckpt_path

def get_image_paths(image_folder):
    possible_image_type = ['jpg', 'png', 'JPEG', 'jpeg']

    image_list = [image_path for image_paths in [glob.glob(os.path.join(image_folder, '*.%s' % ext)) for ext in possible_image_type] for image_path in image_paths]

    return image_list

