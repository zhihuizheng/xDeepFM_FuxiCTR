#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: zhao zhishan
# Created Time : 2021-01-16 04:37
# File Name: fm.py
# Description:
"""
import tensorflow as tf
from fuxictr.estimator.models import BaseModel
from fuxictr.estimator.layers import FM_Layer, EmbeddingLayer


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

        self._task = task
        self._learning_rate = learning_rate
        self._embedding_dim = embedding_dim
        self.build_model()

    def build_model(self):
        embedding_layer = self.embedding_layer
        fm_layer = self.fm_layer
        lr = self._learning_rate

        def model_fn(features, labels, mode, params):
            feature_emb = embedding_layer.forward(features)
            y_pred = fm_layer.forward(features, feature_emb)
            y_pred = tf.squeeze(y_pred)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y_pred))

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions={})

            global_step = tf.train.get_or_create_global_step()
            train_op = tf.train.AdamOptimizer(lr).minimize(self.loss, global_step=global_step)
            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss)
        self.model = tf.estimator.Estimator(model_fn=model_fn)
