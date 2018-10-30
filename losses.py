import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim

tf.app.flags.DEFINE_string('adv_ver', 'ver1', 'ver1 or ver2')
FLAGS = tf.app.flags.FLAGS

VGG_MEAN = [123.68, 116.779, 103.939] # RGB ordered
EPS = 1e-12

class loss_builder:
    def __init__(self):
        self.vgg_path = FLAGS.vgg_path
        self.vgg_used = False
    
    def _rgb_subtraction(self, images):
        channels = tf.split(axis=3, num_or_size_splits=3, value=images)
        for i in range(3):
            channels[i] -= VGG_MEAN[i]
        return tf.concat(axis=3, values=channels)

    def _build_vgg_19(self, images):
        input_images = self._rgb_subtraction(images) / 255.0
        
        ### vgg_19
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=None):
            
            self.conv1_1 = slim.conv2d(input_images, 64, 3, scope='conv1/conv1_1')
            self.conv1_2 = slim.conv2d(self.conv1_1, 64, 3, scope='conv1/conv1_2')
            self.pool1 = slim.max_pool2d(self.conv1_2, 2, scope='pool1')
            
            self.conv2_1 = slim.conv2d(self.pool1, 128, 3, scope='conv2/conv2_1')
            self.conv2_2 = slim.conv2d(self.conv2_1, 128, 3, scope='conv2/conv2_2')
            self.pool2 = slim.max_pool2d(self.conv2_2, 2, scope='pool2')
            
            self.conv3_1 = slim.conv2d(self.pool2, 256, 3, scope='conv3/conv3_1')
            self.conv3_2 = slim.conv2d(self.conv3_1, 256, 3, scope='conv3/conv3_2')
            self.conv3_3 = slim.conv2d(self.conv3_2, 256, 3, scope='conv3/conv3_3')
            self.conv3_4 = slim.conv2d(self.conv3_3, 256, 3, scope='conv3/conv3_4')
            self.pool3 = slim.max_pool2d(self.conv3_4, 2, scope='pool3')

            self.conv4_1 = slim.conv2d(self.pool3, 512, 3, scope='conv4/conv4_1')
            self.conv4_2 = slim.conv2d(self.conv4_1, 512, 3, scope='conv4/conv4_2')
            self.conv4_3 = slim.conv2d(self.conv4_2, 512, 3, scope='conv4/conv4_3')
            self.conv4_4 = slim.conv2d(self.conv4_3, 512, 3, scope='conv4/conv4_4')
            self.pool4 = slim.max_pool2d(self.conv4_4, 2, scope='pool4')

            self.conv5_1 = slim.conv2d(self.pool4, 512, 3, scope='conv5/conv5_1')
            self.conv5_2 = slim.conv2d(self.conv5_1, 512, 3, scope='conv5/conv5_2')
            self.conv5_3 = slim.conv2d(self.conv5_2, 512, 3, scope='conv5/conv5_3')
            self.conv5_4 = slim.conv2d(self.conv5_3, 512, 3, scope='conv5/conv5_4')
            self.pool5 = slim.max_pool2d(self.conv5_4, 2, scope='pool5')
    
        self.vgg_19 = {'conv1_1':self.conv1_1, 'conv1_2':self.conv1_2, 'pool1':self.pool1,
                       'conv2_1':self.conv2_1, 'conv2_2':self.conv2_2, 'pool2':self.pool2,
                       'conv3_1':self.conv3_1, 'conv3_2':self.conv3_2, 'conv3_3':self.conv3_3, 'conv3_4':self.conv3_4, 'pool3':self.pool3,
                       'conv4_1':self.conv4_1, 'conv4_2':self.conv4_2, 'conv4_3':self.conv4_3, 'conv4_4':self.conv4_4, 'pool4':self.pool4,
                       'conv5_1':self.conv5_1, 'conv5_2':self.conv5_2, 'conv5_3':self.conv5_3, 'conv5_4':self.conv5_4, 'pool5':self.pool5,
                       }
        
        self.vgg_used = True
    
    def _lrelu(self, x, a=0.2):
        with tf.name_scope('lrelu'):
            x = tf.identity(x)
            return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

    def _build_discriminator(self, images, is_training):
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training
        }


        with slim.arg_scope([slim.conv2d],
                            activation_fn=None,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            x = slim.conv2d(images, 32, 3, scope='conv1')
            x = self._lrelu(x)
            x = slim.conv2d(x, 32, 3, stride=2, scope='conv2')
            x = self._lrelu(x)
            
            x = slim.conv2d(x, 64, 3, scope='conv3')
            x = self._lrelu(x)
            x = slim.conv2d(x, 64, 3, stride=2, scope='conv4')
            x = self._lrelu(x)

            x = slim.conv2d(x, 128, 3, scope='conv5')
            x = self._lrelu(x)
            x = slim.conv2d(x, 128, 3, stride=2, scope='conv6')
            x = self._lrelu(x)

            x = slim.conv2d(x, 256, 3, scope='conv7')
            x = self._lrelu(x)
            x = slim.conv2d(x, 256, 3, stride=2, scope='conv8')
            x = self._lrelu(x)

            x = slim.conv2d(x, 512, 3, scope='conv9')
            x = self._lrelu(x)
            x = slim.conv2d(x, 512, 3, stride=2, scope='conv10')
            x = self._lrelu(x)

            x = slim.flatten(x, scope='flatten')
            x = slim.fully_connected(x, 1024, activation_fn=None, normalizer_fn=None, scope='fc1')
            x = self._lrelu(x)
            logits = slim.fully_connected(x, 1, activation_fn=None, normalizer_fn=None, scope='fc2')
            outputs = tf.nn.sigmoid(logits)

            return outputs #logits

    def _build_discriminator_ver2(self, images, is_training):
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training
        }


        with slim.arg_scope([slim.conv2d],
                            activation_fn=None,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            
            x = slim.conv2d(images, 64, 3, stride=1, scope='conv1')
            x = self._lrelu(x)
            x = slim.conv2d(x, 64, 3, stride=2, scope='conv2')
            x = self._lrelu(x)
            x = slim.conv2d(x, 128, 3, stride=1, scope='conv3')
            x = self._lrelu(x)
            x = slim.conv2d(x, 128, 3, stride=2, scope='conv4')
            x = self._lrelu(x)
            x = slim.conv2d(x, 256, 3, stride=1, scope='conv5')
            x = self._lrelu(x)
            x = slim.conv2d(x, 256, 3, stride=2, scope='conv6')
            x = self._lrelu(x)
            x = slim.conv2d(x, 512, 3, stride=1, scope='conv7')
            x = self._lrelu(x)
            x = slim.conv2d(x, 512, 3, stride=2, scope='conv8')
            x = self._lrelu(x)
            
            x = slim.flatten(x, scope='flatten')
            x = slim.fully_connected(x, 1024, activation_fn=None, normalizer_fn=None, scope='fc1')
            x = self._lrelu(x)
            logits = slim.fully_connected(x, 1, activation_fn=None, normalizer_fn=None, scope='fc2')
            outputs = tf.nn.sigmoid(logits)

            return outputs

    def _mse(self, gt, pred):
        return tf.losses.mean_squared_error(gt, pred)
    
    def _l1_loss(self, gt, pred):
        return tf.reduce_mean(tf.abs(gt - pred))

    def _gram_matrix(self, features):
        dims = features.get_shape().as_list()
        features = tf.reshape(features, [-1, dims[1] * dims[2], dims[3]])
        
        gram_matrix = tf.matmul(features, features, transpose_a=True)
        normalized_gram_matrix = gram_matrix / (dims[1] * dims[2] * dims[3])

        return normalized_gram_matrix #tf.matmul(features, features, transpose_a=True)

    def _normalize(self, features):
        dims = features.get_shape().as_list()
        return features / (dims[1] * dims[2] * dims[3])
    
    def _preprocess(self, images):
        return (images / 255.0) * 2.0 - 1.0

    def _texture_loss(self, features, patch_size=16):
        '''
        the front part of features : gt features
        the latter part of features : pred features
        I will do calculating gt and pred features at once!
        '''
        #features = self._normalize(features)
        batch_size, h, w, c = features.get_shape().as_list()
        features = tf.space_to_batch_nd(features, [patch_size, patch_size], [[0, 0], [0, 0]])
        features = tf.reshape(features, [patch_size, patch_size, -1, h // patch_size, w // patch_size, c])
        features = tf.transpose(features, [2, 3, 4, 0, 1, 5])
        patches_gt, patches_pred = tf.split(features, 2, axis=0)
        
        patches_gt = tf.reshape(patches_gt, [-1, patch_size, patch_size, c])
        patches_pred = tf.reshape(patches_pred, [-1, patch_size, patch_size, c])

        gram_matrix_gt = self._gram_matrix(patches_gt)
        gram_matrix_pred = self._gram_matrix(patches_pred)
        
        tl_features = tf.reduce_mean(tf.reduce_sum(tf.square(gram_matrix_gt - gram_matrix_pred), axis=-1))
        return tl_features
    
    def _perceptual_loss(self):
        gt_pool5, pred_pool5 = tf.split(self.vgg_19['conv5_4'], 2, axis=0)
        
        pl_pool5 = tf.reduce_mean(tf.reduce_sum(tf.square(gt_pool5 - pred_pool5), axis=-1))
        
        return pl_pool5
   
    def _adv_loss(self, gt_logits, pred_logits):#gan_logits):
        # gt_logits -> real, pred_logits -> fake
        # all values went through tf.nn.sigmoid
        
        
        adv_gen_loss = tf.reduce_mean(-tf.log(pred_logits + EPS))

        adv_disc_loss = tf.reduce_mean(-(tf.log(gt_logits + EPS) + tf.log(1.0 - pred_logits + EPS)))


        #adv_gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_logits, labels=tf.ones_like(pred_logits)))

        #adv_disc_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gt_logits, labels=tf.ones_like(gt_logits)))
        #adv_disc_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_logits, labels=tf.zeros_like(pred_logits)))
        #adv_disc_loss = adv_disc_real + adv_disc_fake


        return adv_gen_loss, adv_disc_loss

    def build_vgg_19(self, gt, pred):
        input_images = tf.concat([gt, pred], axis=0)
        with tf.variable_scope('vgg_19'):
            self._build_vgg_19(input_images)
    
    def build_discriminator(self, gt, pred, is_training=True):
        '''
        build_discriminator is only used for training!
        '''
        
        gt = self._preprocess(gt) # -1.0 ~ 1.0
        pred = self._preprocess(pred) # -1.0 ~ 1.0
        
        
        with tf.variable_scope('discriminator'):
            if FLAGS.adv_ver == 'ver1':
                gt_logits = self._build_discriminator(gt, is_training)
            else:
                gt_logits = self._build_discriminator_ver2(gt, is_training)

        with tf.variable_scope('discriminator', reuse=True):
            b, h, w, c = gt.get_shape().as_list()
            pred.set_shape([b,h,w,c])

            if FLAGS.adv_ver == 'ver1':
                pred_logits = self._build_discriminator(pred, is_training)
            else:
                pred_logits = self._build_discriminator_ver2(pred, is_training)
        
        return gt_logits, pred_logits

    #def build_cycle_discriminator(self, 

    def get_loss(self, gt, pred, type='mse'):
        '''
        'mse', 'inverse_mse', 'fft_mse'
        'perceptual', 'texture'
        'adv, 'cycle_adv'
        '''
        if type == 'mse': # See SRCNN. MSE is very simple loss function.
            gt = self._preprocess(gt)
            pred = self._preprocess(pred)
            return self._mse(gt, pred)
        elif type == 'inverse_mse':
            # gt is the input_lr image!!!
            gt = self._preprocess(gt)
            pred = self._preprocess(pred)
            pred = tf.image.resize_bilinear(pred, size=[tf.shape(gt)[1], tf.shape(gt)[2]])
            return self._mse(gt, pred)
        elif type == 'fft_mse':
            # check whether both gt and pred need preprocessing
            gt = self._preprocess(gt)
            pred = self._preprocess(pred)
            
            ### fft then mse
            gt = tf.cast(gt, tf.complex64)
            pred = tf.cast(pred, tf.complex64)

            gt = tf.fft2d(gt)
            pred = tf.fft2d(pred)

            return self._mse(gt, pred)
        elif type == 'l1_loss':
            gt = self._preprocess(gt)
            pred = self._preprocess(pred)

            return self._l1_loss(gt, pred)
        elif type == 'perceptual': # See Enhancenet.
            if not self.vgg_used:
                self.build_vgg_19(gt, pred)
            
            pl_pool5 = self._perceptual_loss()

            return pl_pool5
        elif type == 'texture': # See Enhancenet, Style transfer papers.
            if not self.vgg_used:
                self.build_vgg_19(gt, pred)
            
            tl_conv1 = self._texture_loss(self.vgg_19['conv1_1'])
            tl_conv2 = self._texture_loss(self.vgg_19['conv2_1'])
            tl_conv3 = self._texture_loss(self.vgg_19['conv3_1'])

            return tl_conv1, tl_conv2, tl_conv3
        elif type == 'adv':
            gt_logits, pred_logits = self.build_discriminator(gt, pred)
            
            adv_gen_loss, adv_disc_loss = self._adv_loss(gt_logits, pred_logits)

            return adv_gen_loss, adv_disc_loss
        else:
            print('%s is not implemented.' % (type))

if __name__ == '__main__':
    
    loss_factory = loss_builder()
    

       
