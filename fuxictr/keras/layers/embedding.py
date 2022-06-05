# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import h5py
import os
import numpy as np
from collections import OrderedDict
# from . import sequence


class EmbeddingLayer(object):
    def __init__(self, 
                 feature_map,
                 embedding_dim,
                 use_pretrain=True,
                 required_feature_columns=[],
                 not_required_feature_columns=[]):
        super(EmbeddingLayer, self).__init__()
        self.embedding_layer = EmbeddingDictLayer(feature_map, 
                                                  embedding_dim,
                                                  use_pretrain=use_pretrain,
                                                  required_feature_columns=required_feature_columns,
                                                  not_required_feature_columns=not_required_feature_columns)

    def forward(self, X):
        feature_emb_dict = self.embedding_layer.forward(X)
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict)
        return feature_emb


class EmbeddingDictLayer(object):
    def __init__(self, 
                 feature_map, 
                 embedding_dim, 
                 use_pretrain=True,
                 required_feature_columns=[],
                 not_required_feature_columns=[]):
        super(EmbeddingDictLayer, self).__init__()
        self._feature_map = feature_map
        self.required_feature_columns = required_feature_columns
        self.not_required_feature_columns = not_required_feature_columns
        self.embedding_layer = OrderedDict()
        for feature, feature_spec in self._feature_map.feature_specs.items():
            if self.is_required(feature):
                if feature_spec["type"] == "numeric":
                    self.embedding_layer[feature] = layers.Embedding(1, embedding_dim, input_length=1)
                elif feature_spec["type"] == "categorical":
                    embedding_matrix = layers.Embedding(feature_spec["vocab_size"], embedding_dim, input_length=1)
                    self.embedding_layer[feature] = embedding_matrix

    def is_required(self, feature):
        """ Check whether feature is required for embedding """
        feature_spec = self._feature_map.feature_specs[feature]
        if len(self.required_feature_columns) > 0 and (feature not in self.required_feature_columns):
            return False
        elif feature in self.not_required_feature_columns:
            return False
        else:
            return True

    def dict2tensor(self, embedding_dict, feature_source=None, feature_type=None):
        # if feature_source is not None:
        #     if not isinstance(feature_source, list):
        #         feature_source = [feature_source]
        #     feature_emb_list = []
        #     for feature, feature_spec in self._feature_map.feature_specs.items():
        #         if feature_spec["source"] in feature_source:
        #             feature_emb_list.append(embedding_dict[feature])
        #     return tf.stack(feature_emb_list, axis=1)
        # elif feature_type is not None:
        #     if not isinstance(feature_type, list):
        #         feature_type = [feature_type]
        #     feature_emb_list = []
        #     for feature, feature_spec in self._feature_map.feature_specs.items():
        #         if feature_spec["type"] in feature_type:
        #             feature_emb_list.append(embedding_dict[feature])
        #     return tf.stack(feature_emb_list, axis=1)
        # else:
        #     return tf.stack(list(embedding_dict.values()), axis=1)
        # return layers.Lambda(lambda x: tf.reduce_sum(x, axis=2))(list(embedding_dict.values()))
        #return keras.backend.stack(list(embedding_dict.values()), axis=1)
        return layers.Concatenate(axis=1)(list(embedding_dict.values()))

    def forward(self, X):
        def slice(x, index):
            return x[:, index]

        feature_emb_dict = OrderedDict()
        for feature, feature_spec in self._feature_map.feature_specs.items():
            if feature in self.embedding_layer:
                if feature_spec["type"] == "numeric":
                    inp = tf.reshape(X[:, feature_spec["index"]], [-1, 1])
                    weights = self.embedding_layer[feature](tf.zeros_like(inp))
                    weights = layers.Lambda(lambda x: keras.backend.squeeze(x, axis=1))(weights)
                    embedding_vec = layers.RepeatVector(1)(layers.Multiply()([inp, weights]))
                elif feature_spec["type"] == "categorical":
                    # inp = tf.cast(X[:, feature_spec["index"]], tf.int32)
                    inp = layers.Lambda(slice, arguments={'index': feature_spec["index"]})(X) #不需要转成int？
                    embedding_vec = layers.RepeatVector(1)(self.embedding_layer[feature](inp))
                feature_emb_dict[feature] = embedding_vec
        return feature_emb_dict


