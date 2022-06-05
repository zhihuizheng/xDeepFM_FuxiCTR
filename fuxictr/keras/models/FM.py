#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: zhao zhishan
# Created Time : 2021-01-16 04:37
# File Name: fm.py
# Description:
"""
import tensorflow as tf
from fuxictr.keras.models import BaseModel
from fuxictr.keras.layers import FM_Layer, EmbeddingLayer
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


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
        # self.model = Sequential()
        # self.model.add(Dense(units=64, activation='relu', input_dim=self._feature_map.num_fields))
        # self.model.add(Dense(units=1, activation='sigmoid'))
        # self.model.compile(loss='binary_crossentropy',
        #                    optimizer='adam',
        #                    metrics=['accuracy'])

        input = keras.Input(shape=(self._feature_map.num_fields,), name="input")
        feature_emb = self.embedding_layer.forward(input)
        feature_emb = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(feature_emb)
        x = layers.Dense(32, activation="relu")(feature_emb)
        output = layers.Dense(1, activation="sigmoid")(x)
        self.model = tf.keras.models.Model(inputs=input, outputs=output)
        self.model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

        self.model.summary()
