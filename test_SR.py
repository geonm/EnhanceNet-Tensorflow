import os
import glob
import sys
import numpy as np
import tensorflow as tf
import sys
import SR_models
import utils
import cv2

tf.app.flags.DEFINE_string('model_path', '/where/your/model/folder', '')
tf.app.flags.DEFINE_string('image_path', '/where/your/test_image/folder', '')
tf.app.flags.DEFINE_string('save_path', '/where/your/generated_image/folder', '')
tf.app.flags.DEFINE_string('run_gpu', '0', '')
FLAGS = tf.app.flags.FLAGS

def load_model(model_path):
    '''
    model_path = '.../where/your/save/model/folder'
    '''
    input_low_images = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='input_low_images')

    model_builder = SR_models.model_builder()

    generated_high_images, resized_low_images = model_builder.generator(input_low_images, is_training=False, model='enhancenet')

    generated_high_images = tf.cast(tf.clip_by_value(generated_high_images, 0, 255), tf.uint8)

    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    gen_vars = [var for var in all_vars if var.name.startswith('generator')]

    saver = tf.train.Saver(gen_vars)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    ckpt_path = utils.get_last_ckpt_path(model_path)
    saver.restore(sess, ckpt_path)

    return input_low_images, generated_high_images, sess

def init_resize_image(im):
    h, w, _ = im.shape
    size = [h, w]
    max_arg = np.argmax(size)
    max_len = size[max_arg]
    min_arg = max_arg - 1
    min_len = size[min_arg]

    maximum_size = 1024
    if max_len < maximum_size:
        maximum_size = max_len
        ratio = 1.0
        return im, ratio
    else:
        ratio = maximum_size / max_len
        max_len = max_len * ratio
        min_len = min_len * ratio
        size[max_arg] = int(max_len)
        size[min_arg] = int(min_len)

        im = cv2.resize(im, (size[1], size[0]))

        return im, ratio

if __name__ == '__main__':
    # set your gpus usage
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.run_gpu
        
    # get pre-trained generator model
    input_image, generated_image, sess = load_model(FLAGS.model_path)

    # get test_image_list
    test_image_list = utils.get_image_paths(FLAGS.image_path)
    
    # make save_folder
    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)

    # do test

    for test_idx, test_image in enumerate(test_image_list):
        loaded_image = cv2.imread(test_image)
        processed_image, tmp_ratio = init_resize_image(loaded_image)

        feed_dict = {input_image : [processed_image[:,:,::-1]]}

        output_image = sess.run(generated_image, feed_dict=feed_dict)

        output_image = output_image[0,:,:,:]

        image_name = os.path.basename(test_image)

        tmp_save_path = os.path.join(FLAGS.save_path, 'SR_' + image_name)
        
        cv2.imwrite(tmp_save_path, output_image[:,:,::-1])

        print('%d / %d completed!!!' % (test_idx + 1, len(test_image_list)))

