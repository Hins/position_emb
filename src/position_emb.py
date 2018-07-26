# -*- coding: utf-8 -*-
# @Time    : 2018/4/8 14:29
# @Author  : panxiaotong
# @Description : calculate word embedding with position information

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
def load_sample(input_file):
    with open(input_file, 'r') as f:
        for item in f:
            elements = item.split(",")
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
                    shape=(cfg.embedding_window, cfg.position_embedding_size),
                    initializer=tf.random_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )

            # [cfg.batch_size, cfg.negative_sample_size + 1, cfg.word_embedding_size]
            word1_embed_init = tf.nn.embedding_lookup(self.word_embed_weight, self.word1)
            print('word1_embed_init shape is %s' % word1_embed_init.get_shape())
            word2_embed_init = tf.nn.embedding_lookup(self.word_embed_weight, self.word2)
            print('word2_embed_init shape is %s' % word2_embed_init.get_shape())
            # [cfg.batch_size, cfg.embedding_window, cfg.position_embedding_size]
            position_embed_init = tf.nn.embedding_lookup(self.position_emb_weight, self.position_samples)
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
            positive_samples = tf.slice(weighted_sim, [0,0], [-1,1])
            negative_samples = tf.slice(weighted_sim, [0,1], [-1,cfg.negative_sample_size + 1])
            self.loss = (-tf.reduce_sum(tf.log(positive_samples), axis=1) + tf.reduce_sum(tf.log(negative_samples), axis=1)) / cfg.batch_size
            self.opt = tf.train.AdamOptimizer().minimize(self.loss)

    def train(self, word1, word2, position):
        return self.sess.run([self.opt, self.loss], feed_dict={
            self.word1: word1,
            self.word2: word2,
            self.position: position})

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("position_emb <input file> <output model>")
        sys.exit()

    load_sample(sys.argv[1])
    if word1_list.shape[0] % (cfg.negative_sample_size + 1) != 0:
        print("samples are incorrect")
        sys.exit()
    total_sample_size = word1_list.shape[0] / (cfg.negative_sample_size + 1)
    total_batch_size = total_sample_size / cfg.batch_size
    train_set_size = total_batch_size * cfg.train_set_ratio

    print('sample_size is %d, train_set_size is %d' % (total_sample_size, train_set_size))

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        PosModelObj = PositionEmbModel(sess)
        tf.global_variables_initializer().run()

        for i in range(train_set_size):
            _, iter_loss = PosModelObj.train(word1_list[i * cfg.batch_size * (cfg.negative_sample_size+1):
                                                        (i + 1) * cfg.batch_size * (cfg.negative_sample_size + 1),:],
                                             word2_list[i * cfg.batch_size * (cfg.negative_sample_size + 1):
                                                        (i + 1) * cfg.batch_size * (cfg.negative_sample_size + 1), :],
                                             position_list[i * cfg.batch_size * (cfg.negative_sample_size + 1):
                                                        (i + 1) * cfg.batch_size * (cfg.negative_sample_size + 1), :])
        sess.close()