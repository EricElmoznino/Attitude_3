import tensorflow as tf
import tensorflow.contrib.data as data
import Helpers as hp
import numpy as np
import shutil
import os
import time


class Model:
    def __init__(self, configuration, image_width, image_height):
        self.conf = configuration

        self.image_shape = [image_width, image_height, 3]
        self.label_shape = [3]

        with tf.variable_scope('hyperparameters'):
            self.keep_prob_placeholder = tf.placeholder(tf.float32, name='dropout_keep_probability')

        self.dataset_placeholders, self.train_dataset, self.iterator = self.create_input_pipeline()
        self.images, self.labels = self.iterator.get_next()
        self.model = self.build_model()
        self.saver = tf.train.Saver()

    def create_input_pipeline(self):
        with tf.variable_scope('input_pipeline'):
            images = tf.placeholder(tf.string, [None, self.conf.seq_size])
            labels = tf.placeholder(tf.float32, [None, self.conf.seq_size] + self.label_shape)
            placeholders = {'images': images, 'labels': labels}

            def process_images(img_files, attitudes):
                img_files = tf.unstack(img_files)
                img_contents = [tf.read_file(f) for f in img_files]
                imgs = [tf.image.decode_jpeg(c, channels=self.image_shape[-1]) for c in img_contents]
                imgs = [tf.divide(tf.cast(i, tf.float32), 255) for i in imgs]
                [i.set_shape(self.image_shape) for i in imgs]
                imgs = tf.stack(imgs)
                return imgs, attitudes

            dataset = data.Dataset.from_tensor_slices((images, labels))
            dataset = dataset.map(process_images)
            dataset = dataset.repeat()
            train_set = dataset.batch(self.conf.batch_size)

            iterator = data.Iterator.from_dataset(train_set)

        return placeholders, train_set, iterator

    def build_model(self):
        with tf.variable_scope('model'):
            image_features = self.extract_image_features()

            encode_decode_state_size = 512
            encoder_outputs = self.encode(image_features, encode_decode_state_size)
            decoder_outputs = self.decode(encoder_outputs, encode_decode_state_size)

            # decoder_outputs = tf.Print(decoder_outputs, [tf.gather_nd(decoder_outputs, [0,0]),tf.gather_nd(decoder_outputs, [0,1])])

            return decoder_outputs

    def extract_image_features(self):
        filter_sizes = [[4, 4], [4, 4], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
        channel_sizes = [20, 40, 40, 80, 80, 160, 160, 320]
        pools = [True, False, True, False, True, False, False, False]

        with tf.variable_scope('image_feature_extraction'):
            features = tf.reshape(self.images, [-1] + self.image_shape)

            with tf.variable_scope('convolution'):
                for i, (filter_size, channel_size, pool) in enumerate(zip(filter_sizes, channel_sizes, pools)):
                    with tf.variable_scope('layer_' + str(i)):
                        features = hp.convolve(features, filter_size, int(features.shape[-1]), channel_size, pad=True)
                        features = tf.nn.relu(features)
                        if pool: features = hp.max_pool(features, [2, 2], pad=True)

            input_size = int(features.shape[1] * features.shape[2] * features.shape[3])
            features = tf.reshape(features, [-1, self.conf.seq_size, input_size])
            features = tf.nn.dropout(features, keep_prob=self.keep_prob_placeholder)

        return features

    def encode(self, image_features, state_size):
        with tf.variable_scope('encoder'):
            cell = tf.contrib.rnn.BasicLSTMCell(state_size)
            encoder_outputs, _ = tf.nn.dynamic_rnn(cell, image_features,
                                                   initial_state=cell.zero_state(batch_size=self.conf.batch_size,
                                                                                 dtype=tf.float32))
        return encoder_outputs

    def decode(self, encoder_outputs, state_size):
        with tf.variable_scope('decoder'):
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=state_size, memory=encoder_outputs)
            cell = tf.contrib.rnn.BasicLSTMCell(state_size)
            cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism)
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell, 3)

            train_helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(
                self.labels,
                tf.constant(self.conf.seq_size, shape=[self.conf.batch_size]),
                0.0
            )
            decoder = tf.contrib.seq2seq.BasicDecoder(cell, train_helper,
                                                      initial_state=cell.zero_state(batch_size=self.conf.batch_size,
                                                                                    dtype=tf.float32))
            decoder_outputs = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=self.conf.seq_size)
            return decoder_outputs[0].rnn_output

    def train(self, train_path):
        with tf.variable_scope('training'):
            sqr_dif = tf.reduce_sum(
                tf.square(
                    tf.reshape(self.model, [-1]+self.label_shape) - tf.reshape(self.labels, [-1]+self.label_shape)
                ),
                1
            )
            mse = tf.reduce_mean(sqr_dif, name='mean_squared_error')
            angle_error = tf.reduce_mean(tf.sqrt(sqr_dif), name='mean_angle_error')
            tf.summary.scalar('angle_error', angle_error)
            optimizer = tf.train.AdamOptimizer().minimize(mse)

        summaries = tf.summary.merge_all()
        if os.path.exists(self.conf.train_log_path):
            shutil.rmtree(self.conf.train_log_path)
        os.mkdir(self.conf.train_log_path)

        print('Starting training\n')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(self.conf.train_log_path, sess.graph)

            start_time = time.time()
            step = 0
            for epoch in range(1, self.conf.epochs + 1):
                epoch_angle_error = 0
                n_samples = self.initialize_iterator(sess, train_path)
                n_batches = int(n_samples / self.conf.batch_size)
                n_steps = n_batches * self.conf.epochs

                for batch in range(n_batches):
                    if step % max(int(n_steps / 1000), 1) == 0:
                        _, a, s = sess.run([optimizer, angle_error, summaries],
                                           feed_dict={self.keep_prob_placeholder: self.conf.keep_prob})
                        train_writer.add_summary(s, step)
                        hp.log_step(step, n_steps, start_time, a)
                    else:
                        _, a = sess.run([optimizer, angle_error],
                                        feed_dict={self.keep_prob_placeholder: self.conf.keep_prob})

                    epoch_angle_error += a
                    step += 1

                hp.log_epoch(epoch, self.conf.epochs, epoch_angle_error / n_batches)

            self.saver.save(sess, os.path.join(self.conf.train_log_path, 'model.ckpt'))

    def initialize_iterator(self, sess, path):
        images, labels = hp.data_at_path(path)
        init = self.iterator.make_initializer(self.train_dataset)
        sess.run(init,
                 feed_dict={self.dataset_placeholders['images']: images,
                            self.dataset_placeholders['labels']: labels})
        return len(images)


