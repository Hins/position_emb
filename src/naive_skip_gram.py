# -*- coding: utf-8 -*-
# @Time        : 2018/8/28 10:56
# @Author      : panxiaotong
# @Description : naive skip-gram for comparison

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
related_dict = {}
word_dictionary_size = 0
def load_sample(input_file, dict_file, related_file):
    global word1_list, word2_list, word_dictionary_size
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
            if elements[2] != "0":    # just user positive samples
                word1_list.append(int(elements[0]))
                word2_list.append(int(elements[1]))
        f.close()

    with open(related_file, 'r') as f:
        for line in f:
            elements = line.strip('\r\n').split(':')
            key = int(elements[0])
            related_dict[key] = {}
            value_list = [int(item) for item in elements[1].split(',') if item is not ""]
            for item in value_list:
                related_dict[key][item] = 1
        f.close()
    word1_list = np.asarray(word1_list)
    word2_list = np.asarray(word2_list)

class SkipGramModel():
    def __init__(self, sess, output_file):
        self.word = tf.placeholder(shape=[cfg.batch_size], dtype=tf.int32)
        self.target = tf.placeholder(shape=[cfg.batch_size], dtype=tf.int32)
        self.pred_label = tf.placeholder(shape=[cfg.batch_size, cfg.negative_sample_size], dtype=tf.int32)
        self.sess = sess
        self.output_file = output_file

        with tf.device('/gpu:0'):
            with tf.variable_scope("skipgram_model"):
                self.word_embed_weight = tf.get_variable(
                    'word_emb',
                    shape=(word_dictionary_size, cfg.word_embedding_size),
                    initializer=tf.random_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )

            # [cfg.batch_size, 1, cfg.word_embedding_size]
            word_embed_init = tf.nn.embedding_lookup(self.word_embed_weight, self.word)
            print('word_embed_init shape is %s' % word_embed_init.get_shape())

            with tf.variable_scope("skipgram_model"):
                proj_weight = tf.get_variable(
                    'proj_layer',
                    shape=(word_dictionary_size, cfg.word_embedding_size),
                    initializer=tf.random_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                print("proj_weight shape is %s" % proj_weight.get_shape())
                proj_bias = tf.get_variable(
                    'proj_bias',
                    shape=(word_dictionary_size),
                    initializer=tf.random_normal_initializer(stddev=cfg.stddev),
                    dtype='float32'
                )
                print("proj_bias shape is %s" % proj_bias.get_shape())
            proj_layer = tf.nn.bias_add(tf.reshape(
                tf.matmul(proj_weight, tf.reshape(word_embed_init, shape=[cfg.word_embedding_size, -1])),
                shape=[-1, word_dictionary_size]
                ), proj_bias)
            # [cfg.batch_size, word_dictionary_size]
            print("proj_layer shape is %s" % proj_layer.get_shape())

            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=proj_weight, biases=proj_bias, labels=tf.reshape(self.target, shape=[-1,1]), inputs=word_embed_init,
                               num_sampled=cfg.negative_sample_size, num_classes=word_dictionary_size))
            self.opt = tf.train.AdamOptimizer().minimize(self.loss)
            self.model = tf.train.Saver()

            softmax_layer = tf.reshape(tf.nn.softmax(logits=proj_layer), shape=[-1])
            print("softmax_layer shape is %s" % softmax_layer.get_shape())
            index_tensor = tf.reshape(tf.concat([tf.reshape(self.target, shape=[-1, 1]), self.pred_label], axis=1), shape=[-1])
            print("index_tensor shape is %s" % index_tensor.get_shape())
            index_score = tf.reshape(tf.nn.embedding_lookup(softmax_layer, index_tensor), shape=[cfg.batch_size, -1])
            print("index_score shape is %s" % index_score.get_shape())
            predict_result = tf.cast(tf.argmax(index_score, axis=1), dtype=tf.int32)
            print("predict_result shape is %s" % predict_result.get_shape())
            comparison = tf.equal(predict_result, np.ones(cfg.batch_size))
            print("comparison shape is %s" % comparison.get_shape())
            self.accuracy = tf.reduce_mean(tf.cast(comparison, dtype=tf.float32))

            self.merged = tf.summary.merge_all()

            with tf.name_scope('Test'):
                self.average_accuracy = tf.placeholder(tf.float32)
                self.accuracy_summary = tf.summary.scalar('accuracy', self.average_accuracy)

            with tf.name_scope('Train'):
                self.average_loss = tf.placeholder(tf.float32)
                self.loss_summary = tf.summary.scalar('average_loss', self.average_loss)

    def train(self, word1, word2):
        return self.sess.run([self.opt, self.loss], feed_dict={
            self.word: word1,
            self.target: word2})

    def validate(self, word1, word2, pred_label):
        return self.sess.run(self.accuracy, feed_dict={
            self.word: word1,
            self.target: word2,
            self.pred_label: pred_label
        })

    def get_loss_summary(self, epoch_loss):
        return self.sess.run(self.loss_summary, feed_dict={self.average_loss: epoch_loss})

    def get_accuracy_summary(self, epoch_accuracy):
        return self.sess.run(self.accuracy_summary, feed_dict={self.average_accuracy: epoch_accuracy})

    def save(self):
        self.model.save(sess, self.output_file)

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("skipgram <input file> <dict file> <related file> <output model>")
        sys.exit()

    load_sample(sys.argv[1], sys.argv[2], sys.argv[3])
    total_sample_size = word1_list.shape[0]
    total_batch_size = total_sample_size / cfg.batch_size
    train_set_size = int(total_batch_size * cfg.train_set_ratio)
    train_set_size_fake = int(total_batch_size * 1)

    labels = np.zeros(shape=[word1_list.shape[0], cfg.negative_sample_size])
    for index, word in enumerate(word1_list):
        accu = 0
        while accu < cfg.negative_sample_size:
            r = random.randint(1, word_dictionary_size)
            if r not in related_dict[word]:
                labels[index][accu] = r
                accu += 1

    print('total_batch_size is %d, train_set_size is %d, word_dictionary_size is %d' %
          (total_batch_size, train_set_size, word_dictionary_size))

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        PosModelObj = SkipGramModel(sess, sys.argv[4])
        tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(cfg.summaries_dir + cfg.train_summary_writer_path, sess.graph)
        test_writer = tf.summary.FileWriter(cfg.summaries_dir + cfg.test_summary_writer_path, sess.graph)

        trainable = False
        for epoch_index in range(cfg.epoch_size):
            loss_sum = 0.0
            for i in range(train_set_size_fake):
                if trainable is True:
                    tf.get_variable_scope().reuse_variables()
                trainable = True
                _, iter_loss = PosModelObj.train(word1_list[i * cfg.batch_size:(i+1) * cfg.batch_size],
                                                 word2_list[i * cfg.batch_size:(i+1) * cfg.batch_size])
                loss_sum += iter_loss
            print("epoch_index %d, loss is %f" % (epoch_index, np.sum(loss_sum) / cfg.batch_size))
            train_loss = PosModelObj.get_loss_summary(np.sum(loss_sum) / cfg.batch_size)
            train_writer.add_summary(train_loss, epoch_index + 1)

            accuracy = 0.0
            for j in range(total_batch_size - train_set_size):
                iter_accuracy = PosModelObj.validate(word1_list[j * cfg.batch_size:(j+1) * cfg.batch_size],
                                                     word2_list[j * cfg.batch_size:(j+1) * cfg.batch_size],
                                                     labels[j * cfg.batch_size:(j+1) * cfg.batch_size])
                accuracy += iter_accuracy
            print("iter %d : accuracy %f" % (epoch_index, accuracy / (total_batch_size - train_set_size)))
            test_accuracy = PosModelObj.get_accuracy_summary(accuracy / (total_batch_size - train_set_size))
            test_writer.add_summary(test_accuracy, epoch_index + 1)
        PosModelObj.save()
        sess.close()