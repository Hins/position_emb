# -*- coding: utf-8 -*-
# @Time        : 2018/7/31 19:14
# @Author      : panxiaotong
# @Description : skip-gram with position embedding information

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
import sys
import tensorflow as tf
from config import cfg
import numpy as np

word1_list = []
word2_list = []
position_list = []
word_dictionary_size = 0
def load_sample(input_file, dict_file):
    global word1_list, word2_list, position_list, word_dictionary_size
    word_dict = {}
    with open(dict_file, 'r') as f:
        for line in f:
            line = line.strip('\r\n').split('\t')
            word_dict[line[0]] = int(line[1])
        f.close()
    word_dictionary_size = len(word_dict)

    with open(input_file, 'r') as f:
        for line in f:
            elements = line.strip('\r\n').split('\t')
            word1_list.append(int(elements[0]))
            word2_list.append(int(elements[1]))
            position_list.append(int(elements[2]))
        f.close()

    word1_list = np.asarray(word1_list)
    word2_list = np.asarray(word2_list)
    position_list = np.asarray(position_list)

class PositionEmbModel():
    def __init__(self, sess):
        self.word = tf.placeholder(shape=[cfg.batch_size, cfg.negative_sample_size + 1], dtype=tf.int32)
        self.target = tf.placeholder(shape=[cfg.batch_size, cfg.negative_sample_size + 1], dtype=tf.int32)
        self.position = tf.placeholder(shape=[cfg.batch_size, cfg.negative_sample_size + 1], dtype=tf.int32)
        self.pred_label = tf.placeholder(shape=[cfg.batch_size * (cfg.negative_sample_size + 1)], dtype=tf.int32)
        self.sess = sess

        with tf.device('/gpu:0'):
            with tf.variable_scope("position_model"):
                self.word_embed_weight = tf.get_variable(
                    'word_emb',
                    shape=(word_dictionary_size, cfg.word_embedding_size),
                    initializer=tf.random_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                self.position_emb_weight = tf.get_variable(
                    'position_emb',
                    shape=(cfg.embedding_window + 1, cfg.position_embedding_size),
                    initializer=tf.random_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )

            # [cfg.batch_size, cfg.negative_sample_size + 1, cfg.word_embedding_size]
            word_embed_init = tf.nn.embedding_lookup(self.word_embed_weight, self.word)
            print('word1_embed_init shape is %s' % word_embed_init.get_shape())
            # [cfg.batch_size, cfg.negative_sample_size + 1, cfg.position_embedding_size]
            position_embed_init = tf.nn.embedding_lookup(self.position_emb_weight, self.position)
            print('pos_embed_init shape is %s' % position_embed_init.get_shape())
            word_position_emb = tf.concat([word_embed_init, position_embed_init], axis=2)
            print("word_position_emb shape is %s" % word_position_emb.get_shape())

            with tf.variable_scope("position_model"):
                proj_weight = tf.get_variable(
                    'proj_layer',
                    shape=(word_dictionary_size, cfg.word_embedding_size),
                    initializer=tf.random_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
            proj_layer = tf.reshape(
                tf.matmul(proj_weight, tf.reshape(word_embed_init, shape=[cfg.word_embedding_size, -1])),
                shape=[-1, word_dictionary_size]
                )
            print("proj_layer shape is %s" % proj_layer.get_shape())
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=proj_layer,
                labels=tf.one_hot(tf.reshape(self.position, shape=[-1,1]), depth=word_dictionary_size)))
            print("loss shape is %s" % self.loss.get_shape())
            self.opt = tf.train.AdamOptimizer().minimize(self.loss)

            predict_result = tf.cast(tf.argmax(proj_layer, axis=1), dtype=tf.int32)
            print("predict_result shape is %s" % predict_result.get_shape())
            comparison = tf.equal(predict_result, self.pred_label)
            print("comparison shape is %s" % comparison.get_shape())
            self.accuracy = tf.reduce_mean(tf.cast(comparison, dtype=tf.float32))

            self.merged = tf.summary.merge_all()

            with tf.name_scope('Test'):
                self.average_accuracy = tf.placeholder(tf.float32)
                self.accuracy_summary = tf.summary.scalar('accuracy', self.average_accuracy)

            with tf.name_scope('Train'):
                self.average_loss = tf.placeholder(tf.float32)
                self.loss_summary = tf.summary.scalar('average_loss', self.average_loss)

    def train(self, word1, word2, position):
        return self.sess.run([self.opt, self.loss], feed_dict={
            self.word: word1,
            self.target: word2,
            self.position: position})

    def validate(self, word1, word2, position, pred_label):
        return self.sess.run(self.accuracy, feed_dict={
            self.word: word1,
            self.target: word2,
            self.position: position,
            self.pred_label: pred_label
        })

    def get_loss_summary(self, epoch_loss):
        return self.sess.run(self.loss_summary, feed_dict={self.average_loss: epoch_loss})

    def get_accuracy_summary(self, epoch_accuracy):
        return self.sess.run(self.accuracy_summary, feed_dict={self.average_accuracy: epoch_accuracy})

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("position_emb <input file> <dict file> <output model>")
        sys.exit()

    load_sample(sys.argv[1], sys.argv[2])
    if word1_list.shape[0] % (cfg.negative_sample_size + 1) != 0:
        print("samples are incorrect")
        sys.exit()
    total_sample_size = word1_list.shape[0] / (cfg.negative_sample_size + 1)
    total_batch_size = total_sample_size / cfg.batch_size
    train_set_size = int(total_batch_size * cfg.train_set_ratio)

    print('total_batch_size is %d, train_set_size is %d, word_dictionary_size is %d' %
          (total_batch_size, train_set_size, word_dictionary_size))

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        PosModelObj = PositionEmbModel(sess)
        tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(cfg.summaries_dir + cfg.train_summary_writer_path, sess.graph)
        test_writer = tf.summary.FileWriter(cfg.summaries_dir + cfg.test_summary_writer_path, sess.graph)

        trainable = False
        for epoch_index in range(cfg.epoch_size):
            loss_sum = 0.0
            for i in range(train_set_size):
                if trainable is True:
                    tf.get_variable_scope().reuse_variables()
                trainable = True
                _, iter_loss = PosModelObj.train(np.reshape(word1_list[i * cfg.batch_size * (cfg.negative_sample_size+1):
                                                (i+1) * cfg.batch_size * (cfg.negative_sample_size + 1)], newshape=[cfg.batch_size, -1]),
                                                 np.reshape(word2_list[i * cfg.batch_size * (cfg.negative_sample_size + 1):
                                                (i+1) * cfg.batch_size * (cfg.negative_sample_size + 1)], newshape=[cfg.batch_size, -1]),
                                                 np.reshape(position_list[i * cfg.batch_size * (cfg.negative_sample_size + 1):
                                                (i+1) * cfg.batch_size * (cfg.negative_sample_size + 1)], newshape=[cfg.batch_size, -1]))
                loss_sum += iter_loss
            print("epoch_index %d, loss is %f" % (epoch_index, np.sum(loss_sum) / cfg.batch_size))
            train_loss = PosModelObj.get_loss_summary(np.sum(loss_sum) / cfg.batch_size)
            train_writer.add_summary(train_loss, epoch_index + 1)

            accuracy = 0.0
            for j in range(total_batch_size - train_set_size):
                word1_validate = np.reshape(word1_list[j * cfg.batch_size * (cfg.negative_sample_size+1):
                                        (j+1) * cfg.batch_size * (cfg.negative_sample_size + 1)], newshape=[cfg.batch_size, -1])
                word2_validate = np.reshape(word2_list[j * cfg.batch_size * (cfg.negative_sample_size + 1):
                                        (j+1) * cfg.batch_size * (cfg.negative_sample_size + 1)], newshape=[cfg.batch_size, -1])
                position_validate = np.reshape(position_list[j * cfg.batch_size * (cfg.negative_sample_size + 1):
                                        (j+1) * cfg.batch_size * (cfg.negative_sample_size + 1)], newshape=[cfg.batch_size, -1])
                for i in range(word1_validate.shape[0]):
                    pos_index = random.randint(0, cfg.negative_sample_size)
                    tmp = word2_validate[i][0]
                    word2_validate[i][0] = word2_validate[i][pos_index]
                    word2_validate[i][pos_index] = tmp
                    tmp = position_validate[i][0]
                    position_validate[i][0] = position_validate[i][pos_index]
                    position_validate[i][pos_index] = tmp
                label_validate = np.reshape(position_validate, newshape=[-1])
                iter_accuracy = PosModelObj.validate(word1_validate, word2_validate, position_validate, label_validate)
                accuracy += iter_accuracy
            print("iter %d : accuracy %f" % (epoch_index, accuracy / (total_batch_size - train_set_size)))
            test_accuracy = PosModelObj.get_accuracy_summary(accuracy / (total_batch_size - train_set_size))
            test_writer.add_summary(test_accuracy, epoch_index + 1)
        sess.close()