#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: zhao zhishan
# Created Time : 2021-01-16 04:37
# File Name: fm.py
# Description:
"""
import tensorflow as tf
from fuxictr.tensorflow.models import BaseModel
from fuxictr.tensorflow.layers import FM_Layer, EmbeddingLayer


class FM(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="FM",
                 gpu=-1,
                 task="binary_classification",
                 learning_rate=1e-3,
                 embedding_dim=10,
                 regularizer=None,
                 **kwargs):
        super(FM, self).__init__(feature_map,
                                 model_id=model_id,
                                 gpu=gpu,
                                 #embedding_regularizer=regularizer,
                                 #net_regularizer=regularizer,
                                 **kwargs)
        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)

        with tf.variable_scope('fm', reuse=tf.AUTO_REUSE):
            self.fm_layer = FM_Layer(feature_map, output_activation=None, #self.get_output_activation(task),
                                     use_bias=True)
        # self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        # self.reset_parameters()
        # self.model_to_device()

        self._task = task
        self._learning_rate = learning_rate
        self._embedding_dim = embedding_dim
        self.build_model()

    def build_model(self):
        self.is_train = tf.placeholder(tf.bool)

        # 或者把整个传进来再拆分
        # self.x = tf.placeholder(tf.float32, shape=[None, self._feature_map.num_fields])
        # self.y = tf.placeholder(tf.float32, shape=[None])
        # # network architecture
        # d1 = tf.layers.dense(self.x, 128, activation=tf.nn.relu, name="dense1")
        # d2 = tf.layers.dense(d1, 1, name="dense2")
        # y_pred = tf.squeeze(d2)

        self.x = tf.placeholder(tf.float32, shape=[None, self._feature_map.num_fields])
        self.y = tf.placeholder(tf.float32, shape=[None])
        feature_emb = self.embedding_layer.forward(self.x) #TODO: 继承什么类就不需要显式调用forward
        y_pred = self.fm_layer.forward(self.x, feature_emb)
        self.y_pred = tf.squeeze(y_pred)

        with tf.name_scope("loss"):
            self.global_step = tf.train.get_or_create_global_step()
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.y_pred))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(self.loss,
                                                                                     global_step=self.global_step)
            correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.saver = tf.train.Saver(max_to_keep=1)

        # with tf.variable_scope('fm', reuse=tf.AUTO_REUSE):
        #     feat_idx = tf.reshape(feat_idx, [-1, field_num])
        #     feat_val = tf.reshape(feat_val, [-1, field_num])
        #     weight_matrix = tf.get_variable('linear_weight',
        #                                     shape=[self._vocab_size],
        #                                     trainable=True)
        #     bias = tf.get_variable('bias',
        #                            shape=[1],
        #                            initializer=tf.zeros_initializer(),
        #                            trainable=True)
        #     embedding_matrix = tf.get_variable('feature_embedding',
        #                                        shape=[self._vocab_size, emb_dim],
        #                                        initializer=tf.uniform_unit_scaling_initializer(),
        #                                        trainable=True)
        #     with tf.device("/cpu:0"):
        #         linear_weight = tf.nn.embedding_lookup(params=weight_matrix, ids=feat_idx)
        #         feat_emb = tf.nn.embedding_lookup(params=embedding_matrix, ids=feat_idx)
        #     # emb regularization
        #     tf.add_to_collection(tf.GraphKeys.WEIGHTS, linear_weight)
        #     tf.add_to_collection(tf.GraphKeys.WEIGHTS, feat_emb)
        #     # first order
        #     first_order_output = tf.reduce_sum(tf.multiply(feat_val, linear_weight), axis=1)
        #     # second order
        #     feat_val = tf.reshape(feat_val, [-1, field_num, 1])
        #     model_input = tf.multiply(feat_emb, feat_val)
        #     square_sum = tf.square(tf.reduce_sum(model_input, axis=1))
        #     sum_square = tf.reduce_sum(tf.square(model_input), axis=1)
        #     second_order_output = tf.reduce_sum(0.5*(tf.subtract(square_sum, sum_square)), axis=1)
        #     # sum
        #     logits = tf.add_n([first_order_output, second_order_output]) + bias
        #     scores = tf.sigmoid(logits)
        #     return logits, scores


# class FM(BaseModel):
#     def __init__(self,
#                  feature_map,
#                  model_id="FM",
#                  gpu=-1,
#                  task="binary_classification",
#                  learning_rate=1e-3,
#                  embedding_dim=10,
#                  regularizer=None,
#                  **kwargs):
#         super(FM, self).__init__(feature_map,
#                                  model_id=model_id,
#                                  gpu=gpu,
#                                  embedding_regularizer=regularizer,
#                                  net_regularizer=regularizer,
#                                  **kwargs)
#         self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
#         self.fm_layer = FM_Layer(feature_map, output_activation=self.get_output_activation(task),
#                                  use_bias=True)
#         self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
#         self.reset_parameters()
#         self.model_to_device()
#
#     def forward(self, inputs):
#         """
#         Inputs: [X, y]
#         """
#         X, y = self.inputs_to_device(inputs)
#         feature_emb = self.embedding_layer(X)
#         y_pred = self.fm_layer(X, feature_emb)
#         return_dict = {"y_true": y, "y_pred": y_pred}
#         return return_dict
