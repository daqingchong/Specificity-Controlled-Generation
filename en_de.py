#  Copyright (c) 2018 Ruqing Zhang. All Rights Reserved.
#  For more information, bug reports, fixes, contact:
#  Ruqing Zhang (daqingchongzrq@gmail.com && zhangruqing@software.ict.ac.cn)


import math
import tensorflow as tf
import numpy as np
import logging
from sklearn.metrics import euclidean_distances

class EncodeDecodeModel:
    SUPPORTED_CELLTYPES = ['lstm', 'gru']
    def __init__(self, cell_type, num_hidden, embedding_size, embedding_topic_size, max_vocab_size_source, max_vocab_size_target,
                 batch_size, grad_clip):
        self.cell_type = cell_type
        self.grad_clip = grad_clip
        self.max_vocab_size_source = max_vocab_size_source
        self.max_vocab_size_target = max_vocab_size_target
        self.embedding_size = embedding_size
        self.embedding_topic_size = embedding_topic_size
        self.num_hidden = num_hidden

        self._check_args()
        self.batch_size = batch_size
        if self.cell_type == 'lstm':
            self.cell_fn = lambda x: tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(x, state_is_tuple=True)
        elif self.cell_type == 'gru':
            self.cell_fn = tf.nn.rnn_cell.GRUCell
        self._create_placeholders()
        self._create_network()

    def _check_args(self):
        if self.cell_type not in self.SUPPORTED_CELLTYPES:
            raise ValueError("This cell type is not supported.")

    def _create_placeholders(self):
        with tf.variable_scope('placeholders'):
        	#batch_size * sentence_num * word_num
            self.encoder_input = tf.placeholder(tf.int32, [None, 30], name='encoder_input' )
            self.encoder_seq_len = tf.placeholder(tf.int32, [None, ], name='encoder_sequence_lengths')

            self.decoder_input = tf.placeholder(tf.int32, [None, 32], name="decoder_input")
            self.decoder_target = tf.placeholder(tf.int32, [None, 32], name="decoder_target")
            self.decoder_weights = tf.placeholder(tf.float32, [None, 32], name="decoder_weights")
            self.decoder_mu = tf.placeholder(tf.float32, [None, 1], name="decoder_mu")



    def _create_word_encoder(self, embedded, sent_length):
        with tf.variable_scope('word_encoder'):
            cell_fw = self.cell_fn(self.num_hidden)
            cell_bw = self.cell_fn(self.num_hidden)

            encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw, cell_bw = cell_bw, dtype=tf.float32,
                                            inputs=embedded, sequence_length=sent_length)

            encoder_output = tf.concat(encoder_output, 2)
            encoder_output = tf.nn.dropout(encoder_output, 0.8)
            encoder_state = encoder_state[1]

        return encoder_output, encoder_state




    def _create_decoder(self, encoder_state, encoder_output, decoder_inputs_embedded, decoder_mu):

        with tf.variable_scope('decoder'):
            cell = self.cell_fn(self.num_hidden)

            w_t = tf.get_variable("proj_w", [self.max_vocab_size_target, self.num_hidden], initializer=tf.contrib.layers.xavier_initializer())
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [self.max_vocab_size_target], initializer=tf.contrib.layers.xavier_initializer())
            output_projection = (w, b)

            decoder_inputs_embedded_trans = tf.transpose(decoder_inputs_embedded, [1, 0, 2])
            embedded = [decoder_inputs_embedded_trans[i] for i in range(32)]

            decoder_outputs, decoder_state = tf.contrib.legacy_seq2seq.attention_decoder(decoder_inputs = embedded, initial_state = encoder_state,
                                                        attention_states = encoder_output,
                                                        cell = cell,
                                                        initial_state_attention = True)

            decoder_outputs = [tf.nn.xw_plus_b(x, w, b) for x in decoder_outputs]

            topic_w = tf.get_variable("topic_w", [self.embedding_topic_size, 1], initializer=tf.contrib.layers.xavier_initializer())
            topic_b = tf.get_variable("topic_b", [1], initializer=tf.contrib.layers.xavier_initializer())
            topic_weight = tf.nn.xw_plus_b(self.embedding_topic, topic_w, topic_b)
            assert topic_weight.get_shape().as_list() == [self.max_vocab_size_target, 1]

            gaussian_mu = tf.tile(decoder_mu, [1, self.max_vocab_size_target])
            assert gaussian_mu.get_shape().as_list() == [None, self.max_vocab_size_target]

            return_topic_weight = tf.transpose(topic_weight)

            topic_weight = tf.nn.sigmoid(topic_weight)
            topic_weight = tf.transpose(topic_weight)
            assert topic_weight.get_shape().as_list() == [1, self.max_vocab_size_target]

            factor = tf.multiply( (gaussian_mu - topic_weight), (gaussian_mu - topic_weight) ) / 2
            assert factor.get_shape().as_list() == [None, self.max_vocab_size_target]
            gaussian_p =  tf.exp(-factor)
            yinsu = math.pow(2 * math.pi,-0.5)
            gaussian_p = gaussian_p * yinsu

            sum_weight1 = tf.get_variable("sum_weight1", [1], initializer=tf.contrib.layers.xavier_initializer())
            sum_weight2 = tf.get_variable("sum_weight2", [1], initializer=tf.contrib.layers.xavier_initializer())
            final_outputs = [ tf.add(x * sum_weight1, gaussian_p * sum_weight2) for x in decoder_outputs]

        def _extract_argmax_and_embed(embedding, output_projection=None, topic_projection=None, weight_projection=None, update_embedding=False):
            def loop_function(prev, _):
                if output_projection is not None:
                    prev_decode = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
                    prev_topic = tf.nn.xw_plus_b(self.embedding_topic, topic_projection[0], topic_projection[1])
                    prev_gaussian_mu = tf.tile(decoder_mu, [1, self.max_vocab_size_target])
                    prev_topic = tf.nn.sigmoid(prev_topic)
                    prev_topic = tf.transpose(prev_topic)
                    prev_factor = - tf.multiply( (prev_gaussian_mu - prev_topic), (prev_gaussian_mu - prev_topic) ) / 2
                    prev_gaussian_p = tf.exp(prev_factor) * math.pow(2 * math.pi,-0.5)

                    prev = prev_decode * weight_projection[0] + prev_gaussian_p * weight_projection[1]

                prev_symbol = tf.argmax(prev, 1)
                emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
                if not update_embedding:
                    emb_prev = tf.stop_gradient(emb_prev)
                return emb_prev
            return loop_function

        with tf.variable_scope('decoder', reuse=True):

            loop_function_predict = _extract_argmax_and_embed(self.embedding_matrix_target, (w, b), (topic_w, topic_b),
                                                                    (sum_weight1, sum_weight2), update_embedding=False)

            decoder_predict_hiddens, _ = tf.contrib.legacy_seq2seq.attention_decoder(decoder_inputs = embedded, initial_state=encoder_state,
                                                                        			cell=cell, attention_states = encoder_output,
                                                                                    loop_function=loop_function_predict,
                                                                                    initial_state_attention = True)

            decoder_predict_hiddens = [tf.nn.xw_plus_b(x, w, b) for x in decoder_predict_hiddens]
            predict_topic_weight = tf.nn.xw_plus_b(self.embedding_topic, topic_w, topic_b)
            predict_gaussian_mu = tf.tile(decoder_mu, [1, self.max_vocab_size_target])
            predict_topic_weight = tf.nn.sigmoid(predict_topic_weight)
            predict_topic_weight = tf.transpose(predict_topic_weight)
            predict_factor = - tf.multiply( (predict_gaussian_mu - predict_topic_weight), (predict_gaussian_mu - predict_topic_weight) ) / 2
            predict_gaussian_p = tf.exp(predict_factor) * math.pow(2 * math.pi,-0.5)
            decoder_predict_logits = [ tf.add(x * sum_weight1, predict_gaussian_p * sum_weight2) for x in decoder_predict_hiddens]

        return final_outputs, decoder_predict_logits, output_projection, sum_weight1, sum_weight2, return_topic_weight




    def _create_network(self):
        with tf.variable_scope('embeddings'):
            initializer = tf.random_uniform_initializer(-0.08, 0.08)
            self.embedding_matrix_source = tf.get_variable("embedding_matrix_source",
                                                    shape = [self.max_vocab_size_source, self.embedding_size],
                                                    initializer = initializer)
            self.embedding_matrix_target = tf.get_variable("embedding_matrix_target",
                                                    shape = [self.max_vocab_size_target, self.embedding_size],
                                                    initializer = initializer)

            #initializer_topic = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            self.embedding_topic = tf.get_variable("embedding_topic",
                                                    shape = [self.max_vocab_size_target, self.embedding_topic_size],
                                                    initializer = initializer)


            encoder_sentences_embedded = tf.nn.embedding_lookup(self.embedding_matrix_source, self.encoder_input)
            decoder_sentences_embedded =  tf.nn.embedding_lookup(self.embedding_matrix_target, self.decoder_input)

        self.word_encoder_outputs, self.word_encoder_states = self._create_word_encoder(encoder_sentences_embedded, self.encoder_seq_len)

        decoder_outputs, decoder_predict_logits, decoder_output_proj, self.weight1, self.weight2, self.topic_weight = self._create_decoder(self.word_encoder_states,
                                                                                            self.word_encoder_outputs,
                                                                                            decoder_sentences_embedded,
                                                                                            self.decoder_mu)

        self.decoder_outputs = decoder_outputs
        self.decoder_outputs = tf.stack(self.decoder_outputs,1)
        assert self.decoder_outputs.get_shape().as_list() == [None, 32, 40004]
        self.loss = tf.contrib.seq2seq.sequence_loss(logits = self.decoder_outputs,
                                                     targets = self.decoder_target,
                                                     weights = self.decoder_weights)


        self.decoder_predict_logits = decoder_predict_logits
        self.decoder_predict = [tf.argmax(logit, 1) for logit in self.decoder_predict_logits]

        optimizer = tf.train.AdamOptimizer()
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.loss, tvars)
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
        self.train_op = optimizer.apply_gradients(zip(clipped_grads, tvars))



    def _fill_feed_dict_train(self, enc_inp, enc_word_num, dec_inp, dec_targ, dec_weight, dec_mu):
        feed_dict = {self.encoder_input: enc_inp,
                     self.encoder_seq_len: enc_word_num}
        feed_dict.update({self.decoder_input: dec_inp})
        feed_dict.update({self.decoder_target: dec_targ })
        feed_dict.update({self.decoder_weights: dec_weight})
        feed_dict.update({self.decoder_mu: dec_mu})
        return feed_dict


    def _fill_feed_dict_predict(self, dec_tar, dec_weight, enc_inp, enc_word_num, dec_mu):
        de = []
        for i in range(self.batch_size):
            de_ = []
            for j in range(32):
                de_.append(40000)
            de.append(de_)
        feed_dict = {self.encoder_input: enc_inp, self.encoder_seq_len: enc_word_num, self.decoder_input: np.array(de)}
        feed_dict.update({self.decoder_mu: dec_mu})
        feed_dict.update({self.decoder_target:dec_tar})
        feed_dict.update({self.decoder_weights: dec_weight})
        return feed_dict


    def train_step(self, enc_inp, enc_word_num, dec_inp, dec_targ, dec_weight, dec_mu):
        feed_dict = self._fill_feed_dict_train(enc_inp, enc_word_num, dec_inp, dec_targ, dec_weight, dec_mu)
        return self.loss, self.train_op, feed_dict


    def predict(self, dec_tar,dec_weight, enc_inp, enc_word_num, dec_mu):
        feed_dict = self._fill_feed_dict_predict(dec_tar,dec_weight, enc_inp, enc_word_num, dec_mu)
        return self.decoder_predict, self.loss, feed_dict

    def get_embedding(self):
        return self.embedding_topic, self.embedding_matrix_target


















