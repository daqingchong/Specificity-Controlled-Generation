# -*- coding:utf-8 -*-
#  Copyright (c) 2018 Ruqing Zhang. All Rights Reserved.
#  For more information, bug reports, fixes, contact:
#  Ruqing Zhang (daqingchongzrq@gmail.com && zhangruqing@software.ict.ac.cn)

import os
import math
import tensorflow as tf
import time
import logging
import numpy as np
import sys
from en_de import EncodeDecodeModel
import random


model_dir = './model'
batch_size = 100

abstract = []
fr_abstract = open('./test/source.txt')
for line in fr_abstract.readlines():
    abstract.append(line.strip('\n').strip().split())
abstract = np.array(abstract)
print abstract.shape


word_num = []
fr_word_num = open('./test/source_word_num.txt')
for line in fr_word_num.readlines():
    word_num.append(line.strip('\n').strip())
word_num = np.array(word_num)
print word_num.shape

target = []
weight = []
fr_target = open('./test/test_target.txt')
for line in fr_target.readlines():
    a = []
    b = []
    for word in line.strip('\n').strip().split():
        a.append(int(word))
        if word == '40002':
            b.append(0)
        else:
             b.append(1)
    target.append(a)
    weight.append(b)
weight = np.array(weight)
target = np.array(target)


vocab = {}
fr_vocab = open('./test/target_vocab.txt')
for line in fr_vocab.readlines():
    arr = line.strip('\n').split()
    vocab[arr[1]] = arr[0]


#contro the value of mu in the range of [0, 1]
dec_mu = []
for i in range(10000):
    tmp = []
    tmp.append(1)
    dec_mu.append(tmp)
dec_mu = np.array(dec_mu)
print dec_mu.shape




model = EncodeDecodeModel(cell_type = 'gru',
                              num_hidden = 300,
                              embedding_size = 300,
                              embedding_topic_size = 50,
                              max_vocab_size_source = 40002,
                              max_vocab_size_target = 40004,
                              batch_size = batch_size,
                              grad_clip = 5.0)

def test():

  config = tf.ConfigProto(log_device_placement=True)
  config.gpu_options.allow_growth = True

  with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
    score = 0
    ckpt = tf.train.get_checkpoint_state(model_dir)

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        usage_embedding, semantic_embedding = model.get_embedding()
        usage, semantic = sess.run([usage_embedding, semantic_embedding])
        usage = np.array(usage)
        np.savetxt('usage_embedding.txt', usage)
        semantic = np.array(semantic)
        np.savetxt('semantic_embedding.txt', semantic)
        '''
        for f in range(abstract.shape[0]/batch_size):
            decoder_targets = target[f*batch_size:(f+1)*batch_size,:]
            encoder_inputs = abstract[f*batch_size:(f+1)*batch_size,:]
            encoder_word_num = word_num[f*batch_size:(f+1)*batch_size]
            decoder_mu = dec_mu[f*batch_size:(f+1)*batch_size,:]
            tit_weights = weight[f*batch_size:(f+1)*batch_size,:]
            print decoder_mu
            decode_predict, loss, feed_dict = model.predict(decoder_targets, tit_weights, encoder_inputs, encoder_word_num, decoder_mu)
            predict, ls = sess.run([decode_predict, loss], feed_dict=feed_dict)
            print ls
            score += float(ls)
            print score
    print math.exp(float(score))
    '''
if __name__ == '__main__':
    test()
