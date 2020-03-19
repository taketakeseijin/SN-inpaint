from layer_sn import *
import tensorflow as tf
Radical = True
LOCAL_SIZE = 64


class Network:
    def __init__(self, x, mask, local_x, is_training, batch_size, alpha_G, start_point):
        self.alpha_G = alpha_G
        self.batch_size = batch_size
        self.imitation = self.generator(x * (1 - mask), is_training)
        self.completion = self.imitation * mask + x * (1 - mask)
        self.local_completion = self.make_completions(
            self.completion, start_point)
        self.real = self.discriminator(x, local_x, reuse=False)
        self.fake = self.discriminator(
            self.completion, self.local_completion, reuse=True)
        self.g_loss = self.calc_g_loss(x, self.completion)
        self.d_loss = self.calc_d_loss(self.real, self.fake)
        if Radical:
            self.mixed_loss = self.calc_mixed_loss(self.fake)
        else:
            self.mixed_loss = self.calc_mixed_loss()
        self.g_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.d_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        """
        self.dl_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator/local')
        self.dg_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator/global')
        self.dc_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator/concatenation')
        """

    def make_completions(self, completion, start):
        size = [1, LOCAL_SIZE, LOCAL_SIZE, 3]
        local_completion = tf.slice(completion, start[0], size)
        for j in range(self.batch_size):
            if j == 0:
                pass
            else:
                local_completion = tf.concat(
                    [local_completion, tf.slice(completion, start[j], size)], 0)
        return local_completion

    def generator(self, x, is_training):
        with tf.variable_scope('generator'):
            with tf.variable_scope('conv1'):
                x = snconv_layer(x, [5, 5, 3, 64], 1)
                x = leaky_relu(x)
            with tf.variable_scope('conv2'):
                x = snconv_layer(x, [3, 3, 64, 128], 2)
                x = leaky_relu(x)
            with tf.variable_scope('conv3'):
                x = snconv_layer(x, [3, 3, 128, 128], 1)
                x = leaky_relu(x)
            with tf.variable_scope('conv4'):
                x = snconv_layer(x, [3, 3, 128, 256], 2)
                x = leaky_relu(x)
            with tf.variable_scope('conv5'):
                x = snconv_layer(x, [3, 3, 256, 256], 1)
                x = leaky_relu(x)
            with tf.variable_scope('conv6'):
                x = snconv_layer(x, [3, 3, 256, 256], 1)
                x = leaky_relu(x)
            with tf.variable_scope('dilated1'):
                x = sndilated_conv_layer(x, [3, 3, 256, 256], 2)
                x = leaky_relu(x)
            with tf.variable_scope('dilated2'):
                x = sndilated_conv_layer(x, [3, 3, 256, 256], 4)
                x = leaky_relu(x)
            with tf.variable_scope('dilated3'):
                x = sndilated_conv_layer(x, [3, 3, 256, 256], 8)
                x = leaky_relu(x)
            with tf.variable_scope('dilated4'):
                x = sndilated_conv_layer(x, [3, 3, 256, 256], 16)
                x = leaky_relu(x)
            with tf.variable_scope('conv7'):
                x = snconv_layer(x, [3, 3, 256, 256], 1)
                x = leaky_relu(x)
            with tf.variable_scope('conv8'):
                x = snconv_layer(x, [3, 3, 256, 256], 1)
                x = leaky_relu(x)
            with tf.variable_scope('deconv1'):
                x = sndeconv_layer(x, [4, 4, 128, 256], [
                                   self.batch_size, 64, 64, 128], 2)
                x = leaky_relu(x)
            with tf.variable_scope('conv9'):
                x = snconv_layer(x, [3, 3, 128, 128], 1)
                x = leaky_relu(x)
            with tf.variable_scope('deconv2'):
                x = sndeconv_layer(x, [4, 4, 64, 128], [
                                   self.batch_size, 128, 128, 64], 2)
                x = leaky_relu(x)
            with tf.variable_scope('conv10'):
                x = snconv_layer(x, [3, 3, 64, 32], 1)
                x = leaky_relu(x)
            with tf.variable_scope('conv11'):
                x = conv_layer(x, [3, 3, 32, 3], 1)
                x = tf.nn.tanh(x)

        return x

    def discriminator(self, global_x, local_x, reuse):
        def global_discriminator(x):
            is_training = tf.constant(True)
            with tf.variable_scope('global'):
                with tf.variable_scope('conv1'):
                    x = snconv_layer(x, [5, 5, 3, 64], 2)
                    x = leaky_relu(x)
                with tf.variable_scope('conv2'):
                    x = snconv_layer(x, [5, 5, 64, 128], 2)
                    x = leaky_relu(x)
                with tf.variable_scope('conv3'):
                    x = snconv_layer(x, [5, 5, 128, 256], 2)
                    x = leaky_relu(x)
                with tf.variable_scope('conv4'):
                    x = snconv_layer(x, [5, 5, 256, 512], 2)
                    x = leaky_relu(x)
                with tf.variable_scope('conv5'):
                    x = snconv_layer(x, [5, 5, 512, 512], 2)
                    x = leaky_relu(x)
                with tf.variable_scope('fc'):
                    x = flatten_layer(x)
                    x = full_connection_layer(x, 1024)
            return x

        def local_discriminator(x):
            is_training = tf.constant(True)
            with tf.variable_scope('local'):
                with tf.variable_scope('conv1'):
                    x = snconv_layer(x, [5, 5, 3, 64], 2)
                    x = leaky_relu(x)
                with tf.variable_scope('conv2'):
                    x = snconv_layer(x, [5, 5, 64, 128], 2)
                    x = leaky_relu(x)
                with tf.variable_scope('conv3'):
                    x = snconv_layer(x, [5, 5, 128, 256], 2)
                    x = leaky_relu(x)
                with tf.variable_scope('conv4'):
                    x = snconv_layer(x, [5, 5, 256, 512], 2)
                    x = leaky_relu(x)
                with tf.variable_scope('fc'):
                    x = flatten_layer(x)
                    x = full_connection_layer(x, 1024)
            return x

        with tf.variable_scope('discriminator', reuse=reuse):
            global_output = global_discriminator(global_x)
            local_output = local_discriminator(local_x)
            with tf.variable_scope('concatenation'):
                output_normal = tf.concat((global_output, local_output), 1)
                output_normal = full_connection_layer(output_normal, 1)
        return output_normal

    def calc_g_loss(self, x, completion):
        loss = tf.nn.l2_loss(x - completion)
        return tf.reduce_mean(loss)

    def calc_d_loss(self, real, fake):
        alpha = 4e-5
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=real, labels=tf.ones_like(real)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake, labels=tf.zeros_like(fake)))
        return tf.add(self.d_loss_real, self.d_loss_fake) * alpha

    if Radical:
        def calc_mixed_loss(self, fake):
            self.d_loss_fake_for_G = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.ones_like(fake)))
            return tf.add(self.alpha_G*self.d_loss_fake_for_G, self.g_loss)
    else:
        def calc_mixed_loss(self):
            return tf.add(-1*self.d_loss, self.g_loss)
