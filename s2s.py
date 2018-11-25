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
batch_size = 128
num_epochs = 15

def readdata(i):
    tmp = str(i)
    file_index = tmp.zfill(2)

    abstract = []
    fr_abstract = open('./data/source'+ file_index)
    for line in fr_abstract.readlines():
        abstract.append(line.strip('\n').strip().split())
    abstract = np.array(abstract)
    print abstract.shape

    title = []
    fr_title = open('./data/title' + file_index)
    for line in fr_title.readlines():
        title.append(line.strip('\n').strip().split())
    title = np.array(title)
    print title.shape

    target = []
    weight = []
    fr_target = open('./data/target' + file_index)
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


    word_num = []
    fr_word_num = open('./data/source_word_num' + file_index)
    for line in fr_word_num.readlines():
        word_num.append(line.strip('\n').strip())
    word_num = np.array(word_num)
    print word_num.shape

    dec_mu = []
    fr_dec_mu = open('./data/dec_mu' + file_index)
    for line in fr_dec_mu.readlines():
        mu = []
        mu.append(line.strip('\n').strip())
        dec_mu.append(mu)
    dec_mu = np.array(dec_mu)
    print dec_mu.shape

    return abstract, title, target, weight, word_num, dec_mu

model = EncodeDecodeModel(cell_type = 'gru',
                              num_hidden = 300,
                              embedding_size = 300,
                              embedding_topic_size = 50,
                              max_vocab_size_source = 40002,
                              max_vocab_size_target = 40004,
                              batch_size = batch_size,
                              grad_clip = 5.0)

def train():

  config = tf.ConfigProto(log_device_placement=True)
  config.gpu_options.allow_growth = True

  with tf.Session(config = config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")

    tmp = 0
    for e in range(num_epochs):
        for file_id in range(37):
            abstract, title, target, weight, word_num, dec_mu = readdata(file_id)
            corpus_num = len(word_num)
            print corpus_num
            init_id = range(corpus_num)
            num_batch = int(corpus_num / batch_size)

            random.shuffle(init_id)
            abstract = abstract[init_id,:]
            title = title[init_id,:]
            target = target[init_id,:]
            weight = weight[init_id,:]
            word_num = word_num[init_id]
            dec_mu = dec_mu[init_id,:]

            for f in range(num_batch):
                assert (f + 1) * batch_size <= corpus_num
                encoder_inputs = abstract[f*batch_size:(f+1)*batch_size,:]
                tit_inputs = title[f*batch_size:(f+1)*batch_size,:]
                targ_inputs = target[f*batch_size:(f+1)*batch_size,:]
                tit_weights = weight[f*batch_size:(f+1)*batch_size,:]
                encoder_word_num = word_num[f*batch_size:(f+1)*batch_size]
                dec_mus = dec_mu[f*batch_size:(f+1)*batch_size,:]

                batch_loss, train_op, feed_dict = model.train_step(encoder_inputs, encoder_word_num, tit_inputs, targ_inputs, tit_weights, dec_mus)
                start_time = time.time()
                sys.stdout.flush()
                batch_loss, train_op = sess.run([batch_loss, train_op], feed_dict=feed_dict)
                batch_perplexity = math.exp(
                    float(batch_loss)) if batch_loss < 300 else float("inf")
                end_time = time.time()
                print(
                        "{}/{} (epoch {}), train_loss = {:.5f},  perplexity = {:.3f}, time/batch = {:.3f}"
                        .format(tmp,
                                num_epochs * num_batch * 37,
                             e, batch_loss, batch_perplexity,  end_time - start_time))
                tmp += 1

        if not os.path.exists(model_dir+str(e)):
            os.makedirs(model_dir+str(e))
        saver.save(sess, os.path.join(model_dir+str(e), 'seq2seq.ckpt'))

if __name__ == '__main__':
    train()
