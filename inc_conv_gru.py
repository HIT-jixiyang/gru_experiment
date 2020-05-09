import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import config as c
from tf_utils import *
class INCEP_ConvGRUCell(object):
    def __init__(self, num_filter, b_h_w,
                 h2h_kernel, i2h_kernel,
                 name, chanel, dtype=tf.float32):
        self._name = name
        self._batch, self._h, self._w = b_h_w
        self._num_filter = num_filter
        self._dtype = dtype
        self._h2h_k = h2h_kernel
        self._i2h_k = i2h_kernel
        self.init_params(chanel)

    @property
    def output_size(self):
        return self._batch, self._h, self._w, self._num_filter

    @property
    def state_size(self):
        return self._batch, self._h, self._w, self._num_filter

    def zero_state(self):
        state_size = self.state_size
        # return tf.Variable(tf.zeros(state_size, dtype=self._dtype), name="init_state", trainable=False)
        return tf.zeros(state_size, dtype=self._dtype)

    def init_params(self, chanel):
        """
        init params for convGRU
        Wi: (kernel, kernel, input_chanel, numfilter*3)
        Wh: (kernel, kernel, numfilter, numfilter*3)
        there will be chanel difference between input and state.
        :param chanel: the chanels of input data
        :return:

        """
        self.Wi=[]
        self._bi=[]
        for i in range(len(self._i2h_k)):
            self.Wi.append(tf.get_variable(name=self._name + "_Wi{}".format(i),
                            shape=self._i2h_k[i],
                            initializer=xavier_initializer(uniform=False),
                            dtype=self._dtype))
            self._bi.append(tf.get_variable(name=self._name + "_bi{}".format(i),
                            shape=(self._num_filter //2*3),
                            initializer=xavier_initializer(uniform=False),
                            dtype=self._dtype))

        self.Wh=[]
        self._bh=[]
        for i in range(len(self._h2h_k)):
            self.Wh.append(tf.get_variable(name=self._name + "_Wh{}".format(i),
                                       shape=self._h2h_k[i],
                                       initializer=xavier_initializer(uniform=False),
                                       dtype=self._dtype))
            self._bh .append(tf.get_variable(name=self._name + "_bh{}".format(i),
                                       shape=(self._num_filter//2* 3),
                                       initializer=xavier_initializer(uniform=False),
                                       dtype=self._dtype))
    def __call__(self, inputs, state):
        """
        do a gru computation
        i2h = leakyRelu(Wi*input + bi)  i2h: (b, h, w, 3*filter)
        h2h = leakyRelu(Wh*state + bh)  h2h: (b. h, w, 3*filter)

        :param inputs: tensor (batch, h, w, c)
        :param state: tensor (batch, h, w, c)
        :return:
        """
        if state is None:
            state = self.zero_state()

        if inputs is not None:
            i2h=inception_conv_2d(inputs,self.Wi,biases=self._bi,name=self._name+'_i2h',strides=[1,1,1,1],)

            # i2h = tf.nn.bias_add(i2h, self._bi)
            # i2h = tf.nn.relu(i2h)
            i2h = tf.split(i2h, 3, axis=3)
        else:

            i2h = None

        h2h = inception_conv_2d(state,self.Wh,biases=self._bh,name=self._name+'_h2h',strides=[1,1,1,1],)

        # h2h = tf.nn.relu(h2h)
        h2h = tf.split(h2h, 3, axis=3)

        if i2h is not None:
            reset_gate = tf.nn.sigmoid(i2h[0] + h2h[0], name=self._name+"_reset")
            update_gate = tf.nn.sigmoid(i2h[1] + h2h[1], name=self._name+"_update")
            new_mem = tf.nn.leaky_relu(i2h[2] + reset_gate * h2h[2],
                                       alpha=0.2, name=self._name+"_leaky")
        else:
            reset_gate = tf.nn.sigmoid(h2h[0], name=self._name + "_reset")
            update_gate = tf.nn.sigmoid(h2h[1], name=self._name + "_update")
            new_mem = tf.nn.leaky_relu(reset_gate * h2h[2],
                                       alpha=0.2, name=self._name + "_leaky")

        next_h = update_gate * state + (1 - update_gate) * new_mem
        self._curr_state = [next_h]
        states = next_h
        output = states
        return output, states
    def unroll(self, length, inputs=None, begin_state=None, merge=True):
        """
        Do gru cycle
        :param length: time length
        :param inputs:  (batch, time_seq, H, W, C)
        :param begin_state:
        :param merge: output a list of tensor or a tensor
        :return:
        outputs:
        """
        if begin_state is None:
            states = self.zero_state()
        else:
            states = begin_state

        outputs = []

        if inputs is not None:
            inputs = tf.unstack(inputs, length, axis=1)
            for i in range(length):
                output, states = self(inputs[i], state=states)
                outputs.append(output)
        else:
            if c.SEQUENCE_MODE:
                inputs = None
                for i in range(length):
                    output, states = self(inputs, state=states)
                    inputs = output
                    outputs.append(output)
            else:
                inputs = [None] * length
                for i in range(length):
                    output, states = self(inputs[i], state=states)
                    outputs.append(output)

        if merge:
            outputs = tf.stack(outputs, axis=1)
        return outputs, states

if __name__ == '__main__':
    # num_filter, b_h_w,
    # h2h_kernel, i2h_kernel,
    # name, chanel, dtype = tf.float32)
    g1=ConvGRUCell(num_filter=c.GEN_ENCODER_GRU_FILTER[0],b_h_w=(2,120,120),h2h_kernel=c.EN_GRU_INCEPTION_H2H[0],i2h_kernel=c.EN_GRU_INCEPTION_I2H[0],name='gru1',chanel=c.GEN_ENCODER_GRU_INCHANEL[0])
    inp=tf.placeholder(shape=[2,120,120,10],dtype=tf.float32)

    print(g1(inp,g1.zero_state()))