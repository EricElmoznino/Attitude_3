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
        self.model = self.build_model_two_channel_deep()
        self.saver = tf.train.Saver()

    def create_input_pipeline(self):
        with tf.variable_scope('input_pipeline'):
            images = tf.placeholder(tf.string, [None, self.conf.seq_size])
            labels = tf.placeholder(tf.float32, [None, self.conf.seq_size] + self.label_shape)
            placeholders = {'images': images, 'labels': labels}

            def process_images(img_files, attitudes):
                # img_contents = tf.map_fn(lambda f: tf.read_file(f), img_files)
                # imgs = tf.map_fn(lambda i: tf.image.decode_jpeg(i, channels=self.image_shape[-1]), img_contents, dtype=tf.uint8)
                # imgs = tf.divide(tf.cast(imgs, tf.float32), 255)
                # imgs.set_shape([seq_size] + self.image_shape)

                img_files = tf.unstack(img_files)
                img_contents = [tf.read_file(f) for f in img_files]
                imgs = [tf.image.decode_jpeg(c, channels=self.image_shape[-1]) for c in img_contents]
                imgs = [tf.divide(tf.cast(i, tf.float32), 255) for i in imgs]
                [i.set_shape(self.image_shape) for i in imgs]
                imgs = tf.stack(imgs)

                # sep = tf.unstack(img_files, axis=0)
                # img_c_l = tf.read_file(sep[0])
                # img_l = tf.image.decode_jpeg(img_c_l, channels=self.image_shape[-1])
                # img_l = tf.divide(tf.cast(img_l, tf.float32), 255)
                # img_l.set_shape(self.image_shape)
                # img_c_r = tf.read_file(sep[1])
                # img_r = tf.image.decode_jpeg(img_c_r, channels=self.image_shape[-1])
                # img_r = tf.divide(tf.cast(img_r, tf.float32), 255)
                # img_r.set_shape(self.image_shape)
                # imgs = tf.stack([img_l, img_r])
                return imgs, attitudes

            dataset = data.Dataset.from_tensor_slices((images, labels))
            dataset = dataset.map(process_images)
            dataset = dataset.repeat()
            train_set = dataset.batch(self.conf.batch_size)

            iterator = data.Iterator.from_dataset(train_set)

        return placeholders, train_set, iterator

    def build_model(self):
        ref_features = self.extract_image_features(self.images_ref)
        new_features = self.extract_image_features(self.images_ref)

    def build_model_two_channel_deep(self):
        filter_sizes = [[4, 4], [4, 4], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
        channel_sizes = [20, 40, 40, 80, 80, 160, 160, 320]
        pools = [True, False, True, False, True, False, False, False]
        fully_connected_sizes = [1024, 1024]
        with tf.variable_scope('model'):
            with tf.variable_scope('convolution'):
                images_ref, images_new = tf.unstack(self.images, axis=1)
                model = tf.concat([images_ref, images_new], axis=3)
                for i, (filter_size, channel_size, pool) in enumerate(zip(filter_sizes, channel_sizes, pools)):
                    with tf.variable_scope('layer_' + str(i)):
                        model = hp.convolve(model, filter_size, int(model.shape[-1]), channel_size, pad=True)
                        model = tf.nn.relu(model)
                        if pool: model = hp.max_pool(model, [2, 2], pad=True)
            with tf.variable_scope('fully_connected'):
                input_size = int(model.shape[1] * model.shape[2] * model.shape[3])
                model = tf.reshape(model, [-1, input_size])
                with tf.variable_scope('layer_1'):
                    weights = hp.weight_variables([input_size, fully_connected_sizes[0]])
                    biases = hp.bias_variables([fully_connected_sizes[0]])
                    model = tf.add(tf.matmul(model, weights), biases)
                    model = tf.nn.relu(model)
                with tf.variable_scope('layer_2'):
                    weights = hp.weight_variables([fully_connected_sizes[0], fully_connected_sizes[1]])
                    biases = hp.bias_variables([fully_connected_sizes[1]])
                    model = tf.add(tf.matmul(model, weights), biases)
                    model = tf.nn.relu(model)
            with tf.variable_scope('output_layer'):
                weights = hp.weight_variables([fully_connected_sizes[-1]] + self.label_shape)
                model = tf.matmul(model, weights)
                model = tf.nn.dropout(model, keep_prob=self.keep_prob_placeholder)
        return model

    def extract_image_features(self, image):
        pass

    def train(self, train_path):
        with tf.variable_scope('training'):
            sqr_dif = tf.reduce_sum(tf.square(self.model - tf.unstack(self.labels, axis=1)[1]), 1)
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


