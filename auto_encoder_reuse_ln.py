import tensorflow as tf
from conv_gru import ConvGRUCell
from tf_utils import *
from tensorflow.contrib.layers import xavier_initializer
import config as c
class Auto_encoder(object):
    def __init__(self, batch, in_seq,
                 out_seq, gru_filters,
                 en_gru_in_chanel, de_gru_in_chanel, conv_kernals, deconv_kernels,
                 conv_strides, deconv_strides, h, w, h2h_kernel, i2h_kernel
                 ):
        self.batch = batch
        self.in_seq = in_seq
        self.out_seq = out_seq
        self.gru_filters = gru_filters
        self.en_gru_in_chanel = en_gru_in_chanel
        self.de_gru_in_chanel = de_gru_in_chanel
        self.conv_kernals_shape = conv_kernals
        self.deconv_kernels_shape = deconv_kernels
        self.conv_kernals = []
        self.deconv_kernels = []
        self.conv_bias = []
        self.deconv_bias = []
        self.conv_strides = conv_strides
        self.deconv_strides = deconv_strides
        self.H = h
        self.W = w
        self.h2h_kernel = h2h_kernel
        self.i2h_kernel = i2h_kernel
        self.encoder_rnn_blocks = []
        self.decoder_rnn_blocks = []
        self.result=[]
        self.en_state_conv_kernel=[]
        self.de_state_conv_kernel=[]
        self.en_state_conv_bias=[]
        self.de_state_conv_bias=[]
        self.resize_map=[]
    def get_input_resize(self,input):
        # for i in range(3):
        h=input.shape.as_list()[1]
        w=input.shape.as_list()[2]
        self.resize_map=[]
        self.resize_map.append(tf.image.resize(input,size=(h//3,w//3)))
        self.resize_map.append(tf.image.resize(input,size=(h//6,w//6)))
        self.resize_map.append(tf.image.resize(input,size=(h//12,w//12)))

    def config_deconv_infer_height(self, height, strides):
        infer_shape = [height]
        for i, s in enumerate(strides[:-1]):
            infer_shape.append(infer_shape[i] // s)
        return infer_shape
    def rnn_encoder(self,in_data):
        self.get_input_resize(in_data)
        for i in range(self.stack_num):
            with tf.name_scope('downsample'):
                conv = conv2d_act(input=in_data,
                                  name=f"Conv{i}",
                                  kernel=self.conv_kernals[i],
                                  bias=self.conv_bias[i],
                                  strides=self.conv_strides[i])
            if c.resize>0:
                with tf.name_scope('resize_concat'):
                    conv=tf.concat([conv,self.resize_map[i]],axis=-1)
            output, states = self.encoder_rnn_blocks[i](inputs=conv,
                                                state=self.rnn_states[i])
            if c.LN:
                with tf.variable_scope('E_LN_{}'.format(i),reuse=tf.AUTO_REUSE):
                # states=tf.contrib.layers.layer_norm(states)
                    if c.multi_ln:

                        states = tf.contrib.layers.layer_norm(LN(states, c.LN_size[i], name='ln'), reuse=tf.AUTO_REUSE, scope='ln')
                    else:
                        states=LN(states,c.LN_size[i])
            in_data = output
            # states=conv2d_act(states, name='state_conv', kernel=self.en_state_conv_kernel[i], bias=self.en_state_conv_bias[i], strides=1, act_type='relu')
            self.rnn_states[i] = states
    def rnn_decoder(self):
        in_data = None
        for i in range(self.stack_num - 1, -1, -1):
            output, states = self.decoder_rnn_blocks[i](inputs=in_data,
                                                state=self.rnn_states[i])
            deconv = deconv2d_act(input=output,
                                  name=f"Deconv{i}",
                                  kernel=self.deconv_kernels[i],
                                  bias=self.deconv_bias[i],
                                  infer_shape=self._infer_shape[i],
                                  strides=self.deconv_strides[i])
            # states = conv2d_act(states, name='state_conv', kernel=self.de_state_conv_kernel[i],
            #                     bias=self.de_state_conv_bias[i], strides=1, act_type='relu')
            if c.LN:
                with tf.variable_scope('D_LN_{}'.format(i), reuse=tf.AUTO_REUSE):
                # states=tf.contrib.layers.layer_norm(states)
                    if c.multi_ln:

                        states = tf.contrib.layers.layer_norm(LN(states, c.LN_size[i], name='ln'), reuse=tf.AUTO_REUSE, scope='ln')
                    else:
                        states=LN(states,c.LN_size[i])

            self.rnn_states[i] = states
            in_data = deconv
        conv_final = tf.nn.conv2d(in_data, self.final_conv[0], strides=(1, 1, 1, 1), padding="SAME",
                                  name="final_conv")
        pred= tf.nn.leaky_relu(tf.nn.bias_add(conv_final, self.final_bias[0]))
        # pred = tf.nn.conv2d(conv_final, filter=self.final_conv[1], strides=(1, 1, 1, 1), padding="SAME",
        #                     name="Pred")
        # pred = tf.nn.leaky_relu(tf.nn.bias_add(pred, self.final_bias[1]))
        # pred = tf.nn.conv2d(pred, filter=self.final_conv[2], strides=(1, 1, 1, 1), padding="SAME",
        #                     name="Pred2")
        # pred = tf.nn.leaky_relu(tf.nn.bias_add(pred, self.final_bias[2]))
        self.result.append(tf.reshape(pred, shape=(self.batch, 1, self.H, self.W, 1)))



    def build_graph(self, in_data):
        feature_map_H = []
        feature_map_W = []
        W_temp = self.W
        H_temp = self.H
        for i in range(len(self.gru_filters)):
            W_temp = W_temp // self.conv_strides[i]
            feature_map_W.append(W_temp)
            H_temp = H_temp // self.conv_strides[i]
            feature_map_H.append(H_temp)
        self.stack_num = len(self.gru_filters)
        for i in range(len(self.gru_filters)):
            self.encoder_rnn_blocks.append(ConvGRUCell(num_filter=self.gru_filters[i],
                                               b_h_w=(self.batch,
                                                      feature_map_H[i],
                                                      feature_map_W[i]),
                                               h2h_kernel=self.h2h_kernel[i],
                                               i2h_kernel=self.i2h_kernel[i],
                                               name="e_cgru_" + str(i),
                                               chanel=self.en_gru_in_chanel[i]))

            self.conv_kernals.append(tf.get_variable(name=f"Conv{i}_W",
                                                     shape=self.conv_kernals_shape[i],
                                                     initializer=xavier_initializer(uniform=False),
                                                     dtype=tf.float32))
            self.conv_bias.append(tf.get_variable(name=f"Conv{i}_b",
                                                  shape=[self.conv_kernals_shape[i][-1]]))
            self.en_state_conv_kernel.append(tf.get_variable(name="EnState_conv{}_W".format(i), shape=[1, 1, self.gru_filters[i], self.gru_filters[i]], initializer=xavier_initializer(uniform=False), dtype=tf.float32))
            self.en_state_conv_bias.append(tf.get_variable(name=f"EnState_conv{i}_b",
                                                           shape=[self.gru_filters[i]]))
            self.de_state_conv_kernel.append(tf.get_variable(name="DeState_conv{}_W".format(i), shape=[1, 1, self.gru_filters[i], self.gru_filters[i]], initializer=xavier_initializer(uniform=False), dtype=tf.float32))
            self.de_state_conv_bias.append(tf.get_variable(name=f"DeState_conv{i}_b",
                                                           shape=[self.gru_filters[i]]))
        self.rnn_states = []
        for block in self.encoder_rnn_blocks:
            self.rnn_states.append(block.zero_state())

        self.deconv_kernels = []
        # self.infer_shape=[]
        self.final_conv = []
        self.final_bias = []
        self._infer_shape = []
        #3*3
        self.final_conv.append(tf.get_variable(name="Final_conv1_W",
                                               shape=(7, 7, c.GEN_DECONV_KERNEL[0][-2], 1),
                                               initializer=xavier_initializer(uniform=False),
                                               dtype=tf.float32))
        self.final_bias.append(tf.get_variable(name="Final_conv1_b",
                                               shape=[1]))
        # self.final_conv.append(tf.get_variable(name="Final_conv2",
        #                                        shape=(1, 1, 8, 2),
        #                                        initializer=xavier_initializer(uniform=False),
        #                                        dtype=tf.float32))
        # self.final_bias.append(tf.get_variable(name="Final_conv2_b",
        #                                        shape=[2]))
        # self.final_conv.append(tf.get_variable(name="Final_conv3",
        #                                        shape=(1, 1, 2, 1),
        #                                        initializer=xavier_initializer(uniform=False),
        #                                        dtype=tf.float32))
        # self.final_bias.append(tf.get_variable(name="Final_conv3_b",
        #                                        shape=[1]))
        self.infer_shape = self.config_deconv_infer_height(self.H, self.deconv_strides)
        for i in range(len(self.gru_filters)):
            self.decoder_rnn_blocks.append(ConvGRUCell(num_filter=self.gru_filters[i],
                                                       b_h_w=(self.batch,
                                                              feature_map_H[i],
                                                              feature_map_W[i]),
                                                       h2h_kernel=self.h2h_kernel[i],
                                                       i2h_kernel=self.i2h_kernel[i],
                                                       name="f_cgru_" + str(i),
                                                       chanel=self.de_gru_in_chanel[i]))
            self.deconv_kernels.append(tf.get_variable(name=f"Deconv{i}_W",
                                                       shape=self.deconv_kernels_shape[i],
                                                       initializer=xavier_initializer(uniform=False),
                                                       dtype=tf.float32))
            self.deconv_bias.append(tf.get_variable(name=f"Deconv{i}_b",
                                                    shape=[self.deconv_kernels_shape[i][-2]]))
            self._infer_shape.append(
                (self.batch, self.infer_shape[i], self.infer_shape[i], self.deconv_kernels_shape[i][-2]))
        with tf.variable_scope("AE", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("Encoder"):
                for i in range(self.in_seq):
                    self.rnn_encoder(in_data[:, i, ...])

            with tf.variable_scope("Forecaster", reuse=tf.AUTO_REUSE):
                for i in range(self.out_seq):
                    self.rnn_decoder()
                self.pred = tf.concat(self.result, axis=1)
                return self.pred




