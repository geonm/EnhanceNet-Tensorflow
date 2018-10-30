import os
import time
import glob
import cv2
import random
import numpy as np
import tensorflow as tf
try:
    import data_util
except ImportError:
    from dataset import data_util

tf.app.flags.DEFINE_float('SR_scale', 4.0, '')
tf.app.flags.DEFINE_boolean('random_resize', False, 'True or False')
tf.app.flags.DEFINE_string('load_mode', 'real', 'real or text') # real -> COCO DB
FLAGS = tf.app.flags.FLAGS

'''
image_path = '/where/your/images/*.jpg'
'''

def load_image(im_fn, hr_size):
    high_image = cv2.imread(im_fn, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)[:,:,::-1] # rgb converted
    
    resize_scale = 1 / FLAGS.SR_scale
    
    '''
    if FLAGS.random_resize:
        resize_table = [0.5, 1.0, 1.5, 2.0]
        selected_scale = np.random.choice(resize_table, 1)[0]
        shrinked_hr_size = int(hr_size / selected_scale)

        h, w, _ = high_image.shape
        if h <= shrinked_hr_size or w <= shrinked_hr_size:
            high_image = cv2.resize(high_image, (hr_size, hr_size))
        else:
            h_edge = h - shrinked_hr_size
            w_edge = w - shrinked_hr_size
            h_start = np.random.randint(low=0, high=h_edge, size=1)[0]
            w_start = np.random.randint(low=0, high=w_edge, size=1)[0]
            high_image_crop = high_image[h_start:h_start+hr_size, w_start:w_start+hr_size, :]
            high_image = cv2.resize(high_image_crop, (hr_size, hr_size))
    '''
    h, w, _ = high_image.shape
    if h <= hr_size or w <= hr_size:
        high_image = cv2.resize(high_image, (hr_size, hr_size),
                                interpolation=cv2.INTER_AREA)
    else:
         h_edge = h - hr_size
         w_edge = w - hr_size
         h_start = np.random.randint(low=0, high=h_edge, size=1)[0]
         w_start = np.random.randint(low=0, high=w_edge, size=1)[0]
         high_image = high_image[h_start:h_start+hr_size, w_start:w_start+hr_size, :]

    low_image = cv2.resize(high_image, (0, 0), fx=resize_scale, fy=resize_scale)
                           #interpolation=cv2.INTER_AREA)
    return high_image, low_image

def load_txtimage(im_fn, hr_size, batch_size):
    high_image = cv2.imread(im_fn, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)[:,:,::-1]
    resize_scale = 1 / FLAGS.SR_scale
    
    lr_size = int(hr_size * resize_scale)

    h, w, _ = high_image.shape
    
    hr_batch = np.zeros([batch_size, hr_size, hr_size, 3], dtype='float32')
    lr_batch = np.zeros([batch_size, lr_size, lr_size, 3], dtype='float32')

    h_edge = h - hr_size
    w_edge = w - hr_size

    passed_idx = 0
    max_iter = 200
    iter_idx = 0
    while passed_idx < batch_size:
        h_start = np.random.randint(low=0, high=h_edge, size=1)[0]
        w_start = np.random.randint(low=0, high=w_edge, size=1)[0]

        crop_hr_image = high_image[h_start:h_start + hr_size, w_start:w_start+hr_size,:]
        
        if np.mean(crop_hr_image) < 250.0:
            hr_batch[passed_idx,:,:,:] = crop_hr_image.copy()
            crop_lr_image = cv2.resize(crop_hr_image, (0, 0), fx=0.25, fy=0.25,
                                       interpolation=cv2.INTER_AREA)
            lr_batch[passed_idx,:,:,:] = crop_lr_image.copy()
            passed_idx += 1
        else:
            iter_idx += 1
        
        if iter_idx == max_iter:
            crop_lr_image = cv2.resize(crop_hr_image, (0, 0), fx=0.25, fy=0.25,
                                       interpolation=cv2.INTER_AREA)
            while passed_idx < batch_size:
                hr_batch[passed_idx,:,:,:] = crop_hr_image.copy()
                lr_batch[passed_idx,:,:,:] = crop_lr_image.copy()
                passed_idx += 1
            return hr_batch, lr_batch

    return hr_batch, lr_batch

def get_record(image_path):
    images = glob.glob(image_path)
    print('%d files found' % (len(images)))

    if len(images) == 0:
        raise FileNotFoundError('check your training dataset path')
    
    index = list(range(len(images)))

    while True:
        random.shuffle(index)
        
        for i in index:
            im_fn = images[i]
            
            yield im_fn #high_image, low_image


def generator(image_path, hr_size=512, batch_size=32):
    high_images = []
    low_images = []
    
    for im_fn in get_record(image_path):
        try:
            # TODO: data augmentation
            '''
            used augmentation methods
            only linear augmenation methods will be used:
            random resize, ...
            not yet implemented

            '''
            if FLAGS.load_mode == 'real':
                high_image, low_image = load_image(im_fn, hr_size)
                
                high_images.append(high_image)
                low_images.append(low_image)
            elif FLAGS.load_mode == 'text':
                high_images, low_images = load_txtimage(im_fn, hr_size, batch_size)

            if len(high_images) == batch_size:
                yield high_images, low_images

                high_images = []
                low_images = []

        except FileNotFoundError as e:
            print(e)
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue

def get_generator(image_path, **kwargs):
    return generator(image_path, **kwargs)

## image_path = '/where/is/your/images/*.jpg'
def get_batch(image_path, num_workers, **kwargs):
    try:
        generator = get_generator(image_path, **kwargs)
        enqueuer = data_util.GeneratorEnqueuer(generator, use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_ouptut = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.001)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()

if __name__ == '__main__':
    image_path = '/data/OCR/DB/icdar_rctw_training/icdar2013_test/*.jpg'
    num_workers = 4
    batch_size = 32
    input_size = 32
    data_generator = get_batch(image_path=image_path,
                               num_workers=num_workers,
                               batch_size=batch_size,
                               hr_size=int(input_size*FLAGS.SR_scale))

    for _ in range(100):
        start_time = time.time()
        data = next(data_generator)
        high_images = np.asarray(data[0])
        low_images = np.asarray(data[1])
        print('%d done!!! %f' % (_, time.time() - start_time), high_images.shape, low_images.shape)
        for sub_idx, (high_image, low_image) in enumerate(zip(high_images, low_images)):
            hr_save_path = '/data/IE/dataset/test_hr/%03d_%02d_hr_image.jpg' % (_, sub_idx)
            lr_save_path = '/data/IE/dataset/test_lr/%03d_%02d_sr_image.jpg' % (_, sub_idx)
            cv2.imwrite(hr_save_path, high_image[:,:,::-1])
            cv2.imwrite(lr_save_path, low_image[:,:,::-1])
