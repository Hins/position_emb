# -*- coding: utf-8 -*-
# @Time    : 2018/4/8 14:29
# @Author  : panxiaotong
# @Description : calculate word similarity with position information

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
import sys
sys.path.append("..")
import tensorflow as tf
from util.config import cfg
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
        self.word1 = tf.placeholder(shape=[cfg.batch_size, cfg.negative_sample_size + 1], dtype=tf.int32)
        self.word2 = tf.placeholder(shape=[cfg.batch_size, cfg.negative_sample_size + 1], dtype=tf.int32)
        self.position = tf.placeholder(shape=[cfg.batch_size, cfg.negative_sample_size + 1], dtype=tf.int32)
        self.pred_label = tf.placeholder(shape=[cfg.batch_size], dtype=tf.int32)
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
            word1_embed_init = tf.nn.embedding_lookup(self.word_embed_weight, self.word1)
            print('word1_embed_init shape is %s' % word1_embed_init.get_shape())
            word2_embed_init = tf.nn.embedding_lookup(self.word_embed_weight, self.word2)
            print('word2_embed_init shape is %s' % word2_embed_init.get_shape())
            # [cfg.batch_size, cfg.embedding_window, cfg.position_embedding_size]
            position_embed_init = tf.nn.embedding_lookup(self.position_emb_weight, self.position)
            print('pos_embed_init shape is %s' % position_embed_init.get_shape())

            # [cfg.batch_size, cfg.negative_sample_size + 1]
            sim_word1_word2 = tf.sigmoid(tf.reshape(tf.reduce_sum(tf.multiply(
                tf.reshape(word1_embed_init, shape=[-1, cfg.word_embedding_size]),
                tf.reshape(word2_embed_init, shape=[-1, cfg.word_embedding_size])), axis=1), shape=[cfg.batch_size, -1]))
            print("sim_word1_word2 shape is %s" % sim_word1_word2.get_shape())

            with tf.variable_scope("position_model"):
                position_embedding_weight = tf.get_variable(
                    'position_emb_weight',
                    shape=[cfg.position_embedding_size, 1],
                    initializer=tf.random_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
            # [cfg.batch_size, cfg.negative_sample_size + 1]
            position_weight = tf.reshape(tf.matmul(
                tf.reshape(position_embed_init, shape=[-1, cfg.position_embedding_size]),
                position_embedding_weight), shape=[cfg.batch_size, -1])
            print("position_weight shape is %s" % position_weight.get_shape())

            weighted_sim = tf.multiply(sim_word1_word2, position_weight)
            print("weighted_sim shape is %s" % weighted_sim.get_shape())
            self.prob = tf.nn.softmax((weighted_sim), axis=1, name="prob")
            print("prob shape is %s" % self.prob.get_shape())

            predict_result = tf.cast(tf.argmax(self.prob, axis=1), dtype=tf.int32)
            print("predict_result shape is %s" % predict_result.get_shape())
            comparison = tf.equal(predict_result, self.pred_label)
            print("comparison shape is %s" % comparison.get_shape())
            self.accuracy = tf.reduce_mean(tf.cast(comparison, dtype=tf.float32))

            positive_samples = tf.slice(self.prob, [0,0], [-1,1])
            print("positive_samples shape is %s" % positive_samples.get_shape())
            negative_samples = tf.slice(self.prob, [0,1], [-1,cfg.negative_sample_size])
            print("negative_samples shape is %s" % negative_samples.get_shape())
            self.loss = (-tf.reduce_sum(tf.log(positive_samples), axis=1) + tf.reduce_sum(tf.log(negative_samples), axis=1)) / cfg.batch_size
            print("loss shape is %s" % self.loss.get_shape())
            self.opt = tf.train.AdamOptimizer().minimize(self.loss)

    def train(self, word1, word2, position):
        return self.sess.run([self.opt, self.loss], feed_dict={
            self.word1: word1,
            self.word2: word2,
            self.position: position})

    def validate(self, word1, word2, position, pred_label):
        return self.sess.run([self.accuracy, self.prob], feed_dict={
            self.word1: word1,
            self.word2: word2,
            self.position: position,
            self.pred_label: pred_label
        })


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

    print('total_batch_size is %d, train_set_size is %d' % (total_batch_size, train_set_size))

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        PosModelObj = PositionEmbModel(sess)
        tf.global_variables_initializer().run()

        trainable = False
        for epoch_index in range(cfg.epoch_size):
            loss_sum = 0.0
            for i in range(train_set_size):
                if trainable == True:
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
                label_validate = np.argmax(np.reshape(position_validate, newshape=[cfg.batch_size, -1]), axis=1)
                iter_accuracy, prob = PosModelObj.validate(word1_validate, word2_validate, position_validate, label_validate)
                accuracy += iter_accuracy
            print("iter %d : accuracy %f" % (epoch_index, accuracy / (total_batch_size - train_set_size)))
        sess.close()