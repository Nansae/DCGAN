import tensorflow as tf
import tensorflow.contrib.slim as slim

class DCGAN(object):
    def __init__(self, width, height, latent_dim):
        
        self.IMG_WIDTH = width
        self.IMG_HEIGHT = height
        self.latent_dim = latent_dim
        self.global_step = tf.Variable(initial_value = 0, name='global_step', trainable=False, collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        self.channels = 3

        self.image = tf.placeholder(tf.float32, [None, self.IMG_HEIGHT, self.IMG_WIDTH, self.channels])
        #self.noise = tf.placeholder(tf.float32, [None, self.IMG_HEIGHT, self.IMG_WIDTH, self.channels])
        self.noise = tf.placeholder(tf.float32, [None, self.latent_dim])

        self.gene = self.generator(self.noise)
        self.real = self.discriminator(self.image)
        self.fake = self.discriminator(self.gene, True)

        self.loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real, labels=tf.ones_like(tf.nn.sigmoid(self.real))))
        self.loss_D_gene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake, labels=tf.zeros_like(tf.nn.sigmoid(self.fake))))

        self.loss_D = self.loss_D_real + self.loss_D_gene
        self.loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake, labels=tf.ones_like(tf.nn.sigmoid(self.fake))))
        
        self.vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        self.sample_data = self.generator(self.noise, is_training=False, reuse=True)

        tf.summary.scalar("train_G", self.loss_G)
        tf.summary.scalar("train_D", self.loss_D)
        # tf.summary.image("generator", tf.cast(GAN.gene*255, tf.uint8))
        # tf.summary.image("random_generator", self.gene)

        print("Done building")

    def generator(self, input, is_training=True, reuse=False):
        with tf.variable_scope('generator', reuse=reuse) as scope:
            batch_norm_params = {'decay': 0.9,
                                 'epsilon': 0.001,
                                 'is_training': is_training,
                                 'scope': 'batch_norm'}
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                activation_fn=tf.nn.leaky_relu,
                                padding='SAME',
                                normalizer_fn=None,
                                normalizer_params=None):

                net = slim.fully_connected(input, 4*4*128, activation_fn=tf.nn.leaky_relu)
                net = tf.reshape(net, shape=[-1, 4, 4, 128])

                net = slim.conv2d_transpose(net, 128, [3,3], [2,2])
                net = slim.conv2d_transpose(net, 64, [3,3], [2,2])
                net = slim.conv2d_transpose(net, 32, [3,3], [2,2])
                net = slim.conv2d_transpose(net, self.channels, [3,3], [2,2])
                net = tf.nn.tanh(net)
                return net

    def discriminator(self, input, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse) as scope:
            batch_norm_params = {'decay': 0.9,
                                 'epsilon': 0.001,
                                 'scope': 'batch_norm'}
            with slim.arg_scope([slim.conv2d],                                
                                padding="SAME",
                                activation_fn=tf.nn.leaky_relu,
                                normalizer_fn=None,
                                normalizer_params=None):

                net = slim.conv2d(input, 32, [3, 3], [2, 2])
                net = slim.conv2d(net, 64, [3, 3], [2, 2])
                net = slim.conv2d(net, 128, [3, 3], [2, 2])
                net = slim.flatten(net)
                #logits = slim.fully_connected(net, 1, activation_fn=tf.nn.sigmoid)
                logits = slim.fully_connected(net, 1, normalizer_fn=None, activation_fn=None)
                return logits