import numpy as np
import tensorflow as tf

from utils import gdl_loss
import config as c
from auto_encoder import Auto_encoder
def generator(in_data,gt_data,batch,in_seq=c.IN_SEQ,out_seq=c.OUT_SEQ,h=c.H,w=c.W,in_c=c.IN_CHANEL,incept=False):

    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        ae=Auto_encoder(batch=c.BATCH_SIZE, in_seq=c.IN_SEQ,
                 out_seq=c.OUT_SEQ, gru_filters=c.GEN_ENCODER_GRU_FILTER,
                 en_gru_in_chanel=c.GEN_ENCODER_GRU_INCHANEL, de_gru_in_chanel=c.GEN_DECODER_GRU_INCHANEL, conv_kernals=c.GEN_CONV_KERNEL, deconv_kernels=c.GEN_DECONV_KERNEL,
                 conv_strides=c.GEN_CONV_STRIDE, deconv_strides=c.GEN_DECONV_STRIDE, h=c.H, w=c.W, h2h_kernel=c.GEN_H2H_KERNEL, i2h_kernel=c.GEN_I2H_KERNEL,incept=incept)
        # global_step = tf.Variable(0, trainable=False)
        pred,pred_logit=ae.build_graph(in_data)
        # pred=tf.nn.tanh(pred)
        return pred,pred_logit,ae.en_feature_map,ae.de_feature_map
