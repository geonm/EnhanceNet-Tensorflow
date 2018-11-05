import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim

tf.app.flags.DEFINE_string('upsample', 'nearest', 'nearest, bilinear, or pixelShuffler')
tf.app.flags.DEFINE_string('model', 'enhancenet', 'for now, only enhancenet supported')
tf.app.flags.DEFINE_string('recon_type', 'residual', 'residual or direct')
tf.app.flags.DEFINE_boolean('use_bn', False, 'for res_block_bn')

FLAGS = tf.app.flags.FLAGS

class model_builder:
    def __init__(self):
        return

    def preprocess(self, images):
        pp_images = images / 255.0
        ## simple mean shift
        pp_images = pp_images * 2.0 - 1.0

        return pp_images
    
    def postprocess(self, images):
        pp_images = ((images + 1.0) / 2.0) * 255.0
        
        return pp_images
    
    def tf_nn_lrelu(self, inputs, a=0.2):
        with tf.name_scope('lrelu'):
            x = tf.identity(inputs)
            return (0.5 * (1.0 + a)) * x + (0.5 * (1.0 - a)) * tf.abs(x)
    
    def tf_nn_prelu(self, inputs, scope):
        # scope like 'prelu_1', 'prelu_2', ...
        with tf.variable_scope(scope):
            alphas = tf.get_variable('alpha', inputs.get_shape()[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
            pos = tf.nn.relu(inputs)
            neg = alphas * (inputs - tf.abs(inputs)) * 0.5

            return pos + neg
    
    def res_block(self, features, out_ch, scope):
        input_features = features
        with tf.variable_scope(scope):
            features = slim.conv2d(input_features, out_ch, 3, activation_fn=tf.nn.relu, normalizer_fn=None)
            features = slim.conv2d(features, out_ch, 3, activation_fn=None, normalizer_fn=None)
            
            return input_features + features
 
    def res_block_bn(self, features, out_ch, is_training, scope): # bn-relu-conv!!!
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training
        }
        
        # input_features already gone through bn-relu
        input_features = features
        with tf.variable_scope(scope):
            features = slim.conv2d(input_features, out_ch, 3, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params)
            features = slim.conv2d(features, out_ch, 3, activation_fn=None, normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params)

            return input_features + features
    
    def phaseShift(self, features, scale, shape_1, shape_2):
        X = tf.reshape(features, shape_1)
        X = tf.transpose(X, [0, 1, 3, 2, 4])

        return tf.reshape(X, shape_2)
    
    def pixelShuffler(self, features, scale=2):
        size = tf.shape(features)
        batch_size = size[0]
        h = size[1]
        w = size[2]
        c = features.get_shape().as_list()[-1]#size[3]

        channel_target = c // (scale * scale)
        channel_factor = c // channel_target

        shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
        shape_2 = [batch_size, h * scale, w * scale, 1]

        input_split = tf.split(axis=3, num_or_size_splits=channel_target, value=features) #features, channel_target, axis=3)
        output = tf.concat([self.phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

        return output

    def upsample(self, features, rate=2):
        if FLAGS.upsample == 'nearest':
            return tf.image.resize_nearest_neighbor(features, size=[rate * tf.shape(features)[1], rate * tf.shape(features)[2]])
        elif FLAGS.upsample == 'bilinear':
            return tf.image.resize_bilinear(features, size=[rate * tf.shape(features)[1], rate * tf.shape(features)[2]])
        else: #pixelShuffler
            return self.pixelShuffler(features, scale=2)

    def recon_image(self, inputs, outputs):
        '''
        LR to HR -> inputs: LR, outputs: HR
        HR to LR -> inputs: HR, outputs: LR
        '''
        resized_inputs = tf.image.resize_bicubic(inputs, size=[tf.shape(outputs)[1], tf.shape(outputs)[2]])
        if FLAGS.recon_type == 'residual':
            recon_outputs = resized_inputs + outputs
        else:
            recon_outputs = outputs
        
        resized_inputs = self.postprocess(resized_inputs)
        resized_inputs = tf.cast(tf.clip_by_value(resized_inputs, 0, 255), tf.uint8)
        #tf.summary.image('4_bicubic image', resized_inputs)

        recon_outputs = self.postprocess(recon_outputs)
        
        return recon_outputs, resized_inputs
    
    ### model part
    '''
    list:
    enhancenet
    '''
    def enhancenet(self, inputs, is_training):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=None):
            
            features = slim.conv2d(inputs, 64, 3, scope='conv1')
            
            for idx in range(10):
                if FLAGS.use_bn:
                    features = self.res_block_bn(features, out_ch=64, is_training=is_training, scope='res_block_bn_%d' % (idx))
                else:
                    features = self.res_block(features, out_ch=64, scope='res_block_%d' % (idx))
            
            features = self.upsample(features)
            features = slim.conv2d(features, 64, 3, scope='conv2')
            
            features = self.upsample(features)
            features = slim.conv2d(features, 64, 3, scope='conv3')
            features = slim.conv2d(features, 64, 3, scope='conv4')
            outputs = slim.conv2d(features, 3, 3, activation_fn=None, scope='conv5')

        return outputs
    
    ########## Let's enhance our method!

    def generator(self, inputs, is_training, model='enhancenet'):
        '''
        LR to HR
        '''

        inputs = self.preprocess(inputs)
        
        with tf.variable_scope('generator'):
            if model == 'enhancenet':
                outputs = self.enhancenet(inputs, is_training)

        outputs, resized_inputs = self.recon_image(inputs, outputs)

        return outputs, resized_inputs
    
### test part

if __name__ == '__main__':
    
    batch_size = 64
    h = 512
    w = 512
    c = 3 # rgb
   
    high_images = np.zeros([batch_size, h, w, c]) # gt
    low_images = np.zeros([batch_size, int(h/4), int(w/4), c])

    input_high_images = tf.placeholder(tf.float32, shape=[batch_size, h, w, c], name='input_high_images')
    input_low_images = tf.placeholder(tf.float32, shape=[batch_size, int(h/4), int(w/4), c], name='input_low_images')

    model_builder = model_builder()
        
    outputs = model_builder.generator(input_low_images)

    print(outputs)

    

