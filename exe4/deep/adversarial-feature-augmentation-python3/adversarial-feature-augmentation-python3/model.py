import tensorflow as tf
import tensorflow.contrib.slim as slim

from utils import lrelu

import numpy as np


class Model(object):
    """
    Reference: Volpi et al., Adversarial Feature Augmentation 
    for Unsupervised Domain Adaptation, CVPR 2018
    
    Author: Riccardo Volpi
    """

    def __init__(self, mode='train_feature_generator', learning_rate=0.0003):

        self.mode = mode
        self.learning_rate = learning_rate
        self.hidden_repr_size = 128

    def feature_extractor(self, images, reuse=False, return_output=False):

        '''
        Takes in input images and gives in output:
        1. logits, in the pretrain phase
        2. features, in the feature generation phase
        '''

        # for SVHN images
        if images.get_shape()[3] == 3:
            images = tf.image.rgb_to_grayscale(images)

        with tf.variable_scope('feature_extractor', reuse=reuse):
            with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='VALID'):
                net = slim.conv2d(images, 64, 5, scope='conv1')
                net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
                net = slim.conv2d(net, 128, 5, scope='conv2')
                net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
                net = tf.contrib.layers.flatten(net)
                net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='fc3')
                net = slim.fully_connected(net, self.hidden_repr_size, activation_fn=tf.tanh,
                                           scope='fc4')  # has to be tanh, as it's more welcomed by GANs

                if (
                        self.mode == 'train_feature_extractor') or return_output:  # in the pretraining phase we need logits, in the feature generation phase we need features
                    net = slim.fully_connected(net, 10, activation_fn=None, scope='fc5')

                return net

    def feature_generator(self, noise, labels, reuse=False):

        '''
        Takes in input noise and labels, and
        generates f(z|y)
        '''

        try:  # just to make it work on different Tensorflow releases
            inputs = tf.concat(1, [noise, tf.cast(labels, tf.float32)])
        except:
            inputs = tf.concat([noise, tf.cast(labels, tf.float32)], 1)

        with tf.variable_scope('feature_generator', reuse=reuse):
            with slim.arg_scope([slim.fully_connected], weights_initializer=tf.contrib.layers.xavier_initializer(),
                                biases_initializer=tf.constant_initializer(0.0)):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                    activation_fn=tf.nn.relu, is_training=(self.mode == 'train_feature_generator')):
                    net = slim.fully_connected(inputs, 1024, activation_fn=tf.nn.relu, scope='sgen_fc1')
                    net = slim.batch_norm(net, scope='sgen_bn1')
                    net = slim.dropout(net, 0.5)  # dropout needs to be always on, do not include "is_training"
                    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='sgen_fc2')
                    net = slim.batch_norm(net, scope='sgen_bn2')
                    net = slim.dropout(net, 0.5)
                    net = slim.fully_connected(net, self.hidden_repr_size, activation_fn=tf.tanh, scope='sgen_feat')
                    return net

    def feature_discriminator(self, features, labels, reuse=False):

        '''
        Used in Step 1 to discriminate between
        real and generated features. Takes in
        input features and associated labels and
        has to discriminate whether they are real
        or generated by feature_generator
        '''

        try:  # just to make it work on different Tensorflow releases
            inputs = tf.concat(1, [features, tf.cast(labels, tf.float32)])
        except:
            inputs = tf.concat([features, tf.cast(labels, tf.float32)], 1)

        with tf.variable_scope('feature_discriminator', reuse=reuse):
            with slim.arg_scope([slim.fully_connected], weights_initializer=tf.contrib.layers.xavier_initializer(),
                                biases_initializer=tf.constant_initializer(0.0)):
                net = slim.fully_connected(inputs, 1024, activation_fn=tf.nn.relu, scope='sdisc_fc1')
                net = slim.fully_connected(net, 1, activation_fn=tf.sigmoid, scope='sdisc_prob')
                return net

    def feature_discriminator_DIFA(self, features, reuse=False):

        '''
        Used in Step 2 to discriminate between
        source and target features. Takes in input
        features and has to discriminate whether
        they are from S or from the encoder we are
        training.
        '''

        with tf.variable_scope('feature_discriminator_DIFA', reuse=reuse):
            with slim.arg_scope([slim.fully_connected], weights_initializer=tf.contrib.layers.xavier_initializer(),
                                biases_initializer=tf.constant_initializer(0.0)):
                net = slim.fully_connected(features, 1024, activation_fn=lrelu, scope='sdisc_DIFA_fc1')
                net = slim.fully_connected(net, 1, activation_fn=tf.sigmoid, scope='sdisc_DIFA_prob')
                return net

    def build_model(self):

        if self.mode == 'train_feature_extractor':

            #################################################################################################################
            # Step 0: training a classifier to classify well some images (source images, if using it for domain adaptation) #
            #################################################################################################################

            self.images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'images')
            self.labels = tf.placeholder(tf.int64, [None], 'labels')

            # extracting logits through feature_extractor (C(E_S) in the paper)
            self.logits = self.feature_extractor(self.images)

            self.pred = tf.argmax(self.logits, 1)
            self.correct_pred = tf.equal(self.pred, self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

            self.loss = slim.losses.sparse_softmax_cross_entropy(self.logits, self.labels)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = slim.learning.create_train_op(self.loss, self.optimizer)

            # summary
            loss_summary = tf.summary.scalar('classification_loss', self.loss)
            src_accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
            self.summary_op = tf.summary.merge([loss_summary, src_accuracy_summary])

        elif self.mode == 'train_feature_generator':

            #######################################################################################################################
            # Step 1: training a feature generator to generate realistic features conditioning on a noise vector and a label code #
            #######################################################################################################################

            self.images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'images')
            self.noise = tf.placeholder(tf.float32, [None, 100], 'noise')
            self.labels = tf.placeholder(tf.int64, [None, 10], 'labels')

            # extracting features associated to our images through extractor (E_S in the paper)
            self.real_features = self.feature_extractor(self.images)

            # features generated by feature_generator (S in the paper)
            self.gen_features = self.feature_generator(self.noise, self.labels)

            # In the following:
            #
            # 'real': related to features extracted from images.
            # 'fake': related to generated features.
            # The minimax game is solved in the Least Squares GAN framework.
            # Ones are related to real features, zeros to generated features. Generator is trained by flipping labels.

            self.logits_real = self.feature_discriminator(self.real_features, self.labels)
            self.logits_fake = self.feature_discriminator(self.gen_features, self.labels, reuse=True)

            self.d_loss_real = tf.reduce_mean(tf.square(self.logits_real - tf.ones_like(self.logits_real)))
            self.d_loss_fake = tf.reduce_mean(tf.square(self.logits_fake - tf.zeros_like(self.logits_fake)))

            self.d_loss = self.d_loss_real + self.d_loss_fake

            self.g_loss = tf.reduce_mean(
                tf.square(self.logits_fake - tf.ones_like(self.logits_fake)))  # flipping labels

            self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.g_optimizer = tf.train.AdamOptimizer(self.learning_rate)

            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if 'feature_discriminator' in var.name]
            g_vars = [var for var in t_vars if 'feature_generator' in var.name]

            with tf.variable_scope('train_op', reuse=False):

                # training feature_discriminator to classify real features as ones and fake features as zeros.
                self.d_train_op = slim.learning.create_train_op(self.d_loss, self.d_optimizer,
                                                                variables_to_train=d_vars)

                # training feature_generator to make feature_discriminator classify fake features as ones, i.e. to generate more realistic features.
                self.g_train_op = slim.learning.create_train_op(self.g_loss, self.g_optimizer,
                                                                variables_to_train=g_vars)

                # summary
            d_loss_summary = tf.summary.scalar('d_loss', self.d_loss)
            g_loss_summary = tf.summary.scalar('g_loss', self.g_loss)
            self.summary_op = tf.summary.merge([d_loss_summary, g_loss_summary])

            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

        elif self.mode == 'train_DIFA':

            ########################################################################################################
            # Step 2: training a feature extractor to map both source and target samples in the same feature space #
            ########################################################################################################

            self.noise = tf.placeholder(tf.float32, [None, 100], 'noise')
            self.labels = tf.placeholder(tf.float32, [None, 10], 'labels')  # to generate features with S
            self.src_images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'svhn_images')
            self.trg_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'mnist_images')

            self.trg_labels = self.feature_extractor(self.trg_images, return_output=True)

            # For testing.......................................................................
            self.trg_labels_gt = tf.placeholder(tf.int64, [None],
                                                'mnist_labels_for_testing')  # ground truth for target, used for testing
            self.trg_pred = tf.argmax(self.trg_labels, 1)
            self.trg_correct_pred = tf.equal(self.trg_pred, self.trg_labels_gt)
            self.trg_accuracy = tf.reduce_mean(tf.cast(self.trg_correct_pred, tf.float32))
            # ..................................................................................

            try:  # try/except due to differences among tf versions
                self.images = tf.concat(0, [tf.image.rgb_to_grayscale(self.src_images), self.trg_images])
            except:
                self.images = tf.concat([tf.image.rgb_to_grayscale(self.src_images), self.trg_images], 0)

            self.features_from_S = self.feature_generator(self.noise, self.labels)
            self.features_from_E = self.feature_extractor(self.images, reuse=True)

            self.logits_real = self.feature_discriminator_DIFA(self.features_from_S)
            self.logits_fake = self.feature_discriminator_DIFA(self.features_from_E, reuse=True)

            # Discriminator loss
            self.d_loss_real = tf.reduce_mean(tf.square(self.logits_real - tf.ones_like(self.logits_real)))
            self.d_loss_fake = tf.reduce_mean(tf.square(self.logits_fake - tf.zeros_like(self.logits_fake)))
            self.d_loss = self.d_loss_real + self.d_loss_fake

            # Encoder loss
            self.e_loss = tf.reduce_mean(tf.square(self.logits_fake - tf.ones_like(self.logits_fake)))

            # Optimizers
            self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.e_optimizer = tf.train.AdamOptimizer(self.learning_rate)

            t_vars = tf.trainable_variables()
            e_vars = [var for var in t_vars if 'feature_extractor' in var.name]
            d_vars = [var for var in t_vars if 'feature_discriminator_DIFA' in var.name]

            # train op
            with tf.variable_scope('training_op', reuse=False):
                self.e_train_op = slim.learning.create_train_op(self.e_loss, self.e_optimizer,
                                                                variables_to_train=e_vars)
                self.d_train_op = slim.learning.create_train_op(self.d_loss, self.d_optimizer,
                                                                variables_to_train=d_vars)

            # summary op
            e_loss_summary = tf.summary.scalar('G_loss', self.e_loss)
            d_loss_summary = tf.summary.scalar('D_loss', self.d_loss)
            self.summary_op = tf.summary.merge([e_loss_summary, d_loss_summary])

            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

        else:
            raise Exception('Unknown mode.')