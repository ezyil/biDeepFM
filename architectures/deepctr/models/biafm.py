# -*- coding:utf-8 -*-
"""
Author:
    Ezgi Yildirim, yildirimez@itu.edu.tr

Reference:
    [1] Yıldırım, Ezgi, Payam Azad, and Şule Gündüz Öğüdücü. "biDeepFM: A multi-objective deep factorization machine for reciprocal recommendation." Engineering Science and Technology, an International Journal (2021).

"""

import tensorflow as tf

from ..input_embedding import preprocess_input_embedding, get_linear_logit
from ..layers.core import PredictionLayer
from ..layers.interaction import AFMLayer, FM
from ..layers.utils import concat_fun
from ..utils import check_feature_config_dict


def biAFM(feature_dim_dict, embedding_size=8, use_attention=True, attention_factor=8,
        l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_att=1e-5, afm_dropout=0, init_std=0.0001, seed=1024,
        task='binary', ):
    """Instantiates the Multi-Objective Attentional Factorization Machine architecture.

    :param feature_dim_dict: dict,to indicate sparse field and dense field like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}
    :param embedding_size: positive integer,sparse feature embedding_size
    :param use_attention: bool,whether use attention or not,if set to ``False``.it is the same as **standard Factorization Machine**
    :param attention_factor: positive integer,units in attention net
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_att: float. L2 regularizer strength applied to attention net
    :param afm_dropout: float in [0,1), Fraction of the attention net output units to dropout.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    check_feature_config_dict(feature_dim_dict)

    deep_emb_list, linear_emb_list, dense_input_dict, inputs_list = preprocess_input_embedding(feature_dim_dict,
                                                                                               embedding_size,
                                                                                               l2_reg_embedding,
                                                                                               l2_reg_linear, init_std,
                                                                                               seed,
                                                                                               create_linear_weight=True)

    linear_logit = get_linear_logit(linear_emb_list, dense_input_dict, l2_reg_linear)
    linear_logit_1 = get_linear_logit(linear_emb_list, dense_input_dict, l2_reg_linear)

    fm_input = concat_fun(deep_emb_list, axis=1)
    if use_attention:
        fm_logit = AFMLayer(attention_factor, l2_reg_att, afm_dropout, seed)(deep_emb_list,)
        fm_logit_1 = AFMLayer(attention_factor, l2_reg_att, afm_dropout, seed)(deep_emb_list,)
    else:
        fm_logit = FM()(fm_input)
        fm_logit_1 = FM()(fm_input)

    final_logit = tf.keras.layers.add([linear_logit, fm_logit])
    final_logit_1 = tf.keras.layers.add([linear_logit_1, fm_logit_1])
    output = PredictionLayer(task)(final_logit)
    output_1 = PredictionLayer(task)(final_logit_1)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=[output, output_1])
    return model
