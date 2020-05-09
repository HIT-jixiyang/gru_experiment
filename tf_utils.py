import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
# from config import c

def LN(tensor,size=[24]):
    for s in size:
        n=tensor.shape.as_list()[1]//s
        if tensor.shape.as_list()[1]==s:
            return tf.contrib.layers.layer_norm(tensor)
        list_k=[]
        for k in range(n):
            list_j=[]
            for j in range(n):
                start_x=k*s
                start_y=j*s
                list_j.append(tf.contrib.layers.layer_norm(tensor[:,start_x:start_x+s,start_y:start_y+s,:]))
            list_k.append(tf.concat(list_j,axis=2))
        tensor=tf.concat(list_k,axis=1)
    return tensor

# def LN(tensor,size=[24],name=''):
#     for s in size:
#         n=tensor.shape.as_list()[1]//s
#         if tensor.shape.as_list()[1]==s:
#             return tf.contrib.layers.layer_norm(tensor,reuse=tf.AUTO_REUSE,scope=name)
#         list_k=[]
#         with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
#             for k in range(n):
#                 list_j=[]
#                 for j in range(n):
#                     start_x=k*s
#                     start_y=j*s
#                     list_j.append(tf.contrib.layers.layer_norm(tensor[:,start_x:start_x+s,start_y:start_y+s,:],scope=name+str(k)+str(j),reuse=tf.AUTO_REUSE))
#                 list_k.append(tf.concat(list_j,axis=2))
#         tensor=tf.concat(list_k,axis=1)
#     return tensor


def hinge_loss(x,y,real_hist):
    threshold=[15.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0,55.0,60.0,65.0,70.0,75.0]
    result=[]
    for i in range(len(threshold)):
        m1=tf.maximum(tf.zeros_like(x,dtype=tf.float32), tf.ones_like(x,dtype=tf.float32)*(threshold[i]+1) - x)
        m2=tf.maximum(tf.zeros_like(y,dtype=tf.float32),tf.ones_like(y,dtype=tf.float32)*(threshold[i]+1)-y)

        result.append(tf.abs(tf.reduce_sum(1-m1)-tf.reduce_sum(1-m2))/(real_hist[i]*threshold[i]))
    return tf.reduce_mean(tf.stack(result))
#18241:1145782   7098:1268026 553:1295062
def conv2d(input, name, kshape, strides=(1, 1, 1, 1), dtype=np.float32, padding="SAME"):
    with tf.name_scope(name):
        W = tf.get_variable(name='w_'+name,
                            shape=kshape,
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                            dtype=dtype)
        b = tf.get_variable(name='b_' + name,
                            shape=[kshape[3]],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                            dtype=dtype)
        out = tf.nn.conv2d(input, W, strides=strides, padding=padding)
        out = tf.nn.bias_add(out, b)
        out = tf.nn.leaky_relu(out, alpha=0.2)
        return out


def conv2d_act(input, name, kernel, bias, strides, act_type="leaky", padding="SAME"):
    """
    :param input:
    :param name:
    :param kernel:
    :param bias:
    :param strides:
    :param act_type:
    :param padding:
    :return:
    """
    with tf.name_scope(name):
        input_size = input.shape.as_list()
        if len(input_size) == 5:
            input = tf.reshape(input, shape=(input_size[0] * input_size[1],
                                             input_size[2],
                                             input_size[3],
                                             input_size[4]))

        out = tf.nn.conv2d(input, kernel, strides=(1, strides, strides, 1),
                           padding=padding, name=name)
        out = tf.nn.bias_add(out, bias)
        if act_type == "relu":
            out = tf.nn.relu(out)
        elif act_type == "leaky":
            out = tf.nn.leaky_relu(out, alpha=0.2)

        if len(input_size) == 5:
            out_size = out.shape.as_list()
            out = tf.reshape(out, shape=(input_size[0],
                                         input_size[1],
                                         out_size[-3],
                                         out_size[-2],
                                         out_size[-1]))
        return out


def deconv2d_act(input, name, kernel, bias, infer_shape, strides, act_type="leaky", padding="SAME"):
    with tf.name_scope(name):
        input_size = input.shape.as_list()
        if len(input_size) == 5:
            input = tf.reshape(input, shape=(input_size[0] * input_size[1],
                                             input_size[2],
                                             input_size[3],
                                             input_size[4]))
        out = tf.nn.conv2d_transpose(input, kernel, infer_shape, strides=(1, strides, strides, 1), name=name, padding=padding)
        out = tf.nn.bias_add(out, bias)
        if act_type == "relu":
            out = tf.nn.relu(out)
        elif act_type == "leaky":
            out = tf.nn.leaky_relu(out, alpha=0.2)
        if len(input_size) == 5:
            out_size = out.shape.as_list()
            out = tf.reshape(out, shape=(input_size[0],
                                         input_size[1],
                                         out_size[-3],
                                         out_size[-2],
                                         out_size[-1]))
        return out

def deconv2d(input, name, kshape, n_outputs, strides=(1, 1)):
    with tf.name_scope(name):
        out = tf.contrib.layers.conv2d_transpose(input,
                                                 num_outputs= n_outputs,
                                                 kernel_size=kshape,
                                                 stride=strides,
                                                 padding='SAME',
                                                 weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                                                 biases_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                                 activation_fn=tf.nn.relu)
        return out





def maxpool2d(x,name,kshape=(1, 2, 2, 1), strides=(1, 2, 2, 1)):
    with tf.name_scope(name):
        out = tf.nn.max_pool(x,
                             ksize=kshape, #size of window
                             strides=strides,
                             padding='SAME')
        return out


def upsample(input, name, factor=(2, 2)):
    size = [int(input.shape[1] * factor[0]), int(input.shape[2] * factor[1])]
    with tf.name_scope(name):
        out = tf.image.resize_bilinear(input, size=size, align_corners=None, name=None)
        return out


def fullyConnected(input, name, output_size, dtype=np.float32):
    with tf.name_scope(name):
        input_size = input.shape[1:]
        input_size = int(np.prod(input_size))
        W = tf.get_variable(name='w_'+name,
                            shape=[input_size, output_size],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                            dtype=dtype)
        b = tf.get_variable(name='b_'+name,
                            shape=[output_size],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                            dtype=dtype)
        input = tf.reshape(input, [-1, input_size])
        out = tf.nn.relu(tf.add(tf.matmul(input, W), b))
        return out


def dropout(input, name, keep_rate):
    with tf.name_scope(name):
        out = tf.nn.dropout(input, keep_rate)
        return out






def down_sampling(input, kshape, stride, num_filters, name, padding="SAME"):
    input_size = input.shape.as_list()
    if len(input_size) == 5:
        input = tf.reshape(input, shape=(input_size[0]*input_size[1],
                                         input_size[2],
                                         input_size[3],
                                         input_size[4]))
    out = conv2d_act(input,
                    kernel=kshape,
                    strides=stride,
                    num_filters=num_filters,
                    padding=padding,
                    name=name)
    if len(input_size) == 5:
        out_size = out.shape.as_list()
        out = tf.reshape(out, shape=(input_size[0],
                                     input_size[1],
                                     out_size[-3],
                                     out_size[-2],
                                     out_size[-1]))
    return out


def up_sampling(input, kshape, stride, num_filter, name):

    input_size = input.shape.as_list()
    if len(input_size) == 5:
        input = tf.reshape(input, shape=(input_size[0] * input_size[1],
                                         input_size[2],
                                         input_size[3],
                                         input_size[4]))
    out = deconv2d_act(input,
                        kernel=kshape,
                        stride=stride,
                        num_filters=num_filter,
                        name=name)

    if len(input_size) == 5:
        out_size = out.shape.as_list()
        out = tf.reshape(out, shape=(input_size[0],
                                     input_size[1],
                                     out_size[-3],
                                     out_size[-2],
                                     out_size[-1]))
    return out

def _weights(name, shape, mean=0.0, stddev=0.02):
  """ Helper to create an initialized Variable
  Args:
    name: name of the variable
    shape: list of ints
    mean: mean of a Gaussian
    stddev: standard deviation of a Gaussian
  Returns:
    A trainable variable
  """
  var = tf.get_variable(
    name, shape,
    initializer=tf.random_normal_initializer(
      mean=mean, stddev=stddev, dtype=tf.float32))
  return var

def _biases(name, shape, constant=0.0):
  """ Helper to create an initialized Bias with constant
  """
  return tf.get_variable(name, shape,
            initializer=tf.constant_initializer(constant))

def _instance_norm(input,name=''):
  """ Instance Normalization
  """
  with tf.variable_scope("instance_norm_"+name):
    depth = input.get_shape()[3]
    scale = _weights("scale", [depth], mean=1.0)
    offset = _biases("offset", [depth])
    mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (input-mean)*inv
    return scale*normalized + offset

def max_min_norm(x, filters=None, epsilon=1e-6, scope=None, reuse=None):
    if filters is None:
        filters = x.get_shape()[-1]
    with tf.variable_scope(scope, default_name="max_min_norm", values=[x], reuse=None):
        scale = tf.get_variable(
            "max_min_norm_scale", [filters], regularizer=None, initializer=tf.ones_initializer())
        bias = tf.get_variable(
            "max_min_norm_bias", [filters], regularizer=None, initializer=tf.zeros_initializer())
        result = max_min_norm_python(x, epsilon, scale, bias)
        return result
def max_min_norm_python(x, epsilon, scale, bias):
    max = tf.reduce_max(x, axis=[-1], keep_dims=True)
    min = tf.reduce_min(x, axis=[-1], keep_dims=True)
    # variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
    norm_x = (x - min) /(max-min+epsilon)
    return norm_x * scale + bias

def max_pool_loss(pred,gt):
    shape=pred.shape.as_list()
    pred=tf.reshape(pred,[shape[0]*shape[1],shape[2],shape[3],shape[4]])
    gt=tf.reshape(gt,[shape[0]*shape[1],shape[2],shape[3],shape[4]])
    p_max_pool5 = maxpool2d(pred, name='pred_max_pool_5', kshape=(1, shape[2] // 18, shape[2] // 18, 1),
                            strides=(1, 2, 2, 1))
    gt_max_pool5 = maxpool2d(gt, name='gt_max_pool_5', kshape=(1, shape[2] // 18, shape[2] // 18, 1),
                             strides=(1, 2, 2, 1))
    l5 = tf.reduce_mean(tf.abs(p_max_pool5 - gt_max_pool5))
    p_max_pool4 = maxpool2d(pred, name='pred_max_pool_4', kshape=(1, shape[2] // 9, shape[2] // 9, 1),
                            strides=(1, shape[2] // 18, shape[2] // 18, 1))
    gt_max_pool4 = maxpool2d(gt, name='gt_max_pool_4', kshape=(1, shape[2] // 9, shape[2] // 9, 1),
                             strides=(1, shape[2] // 18, shape[2] // 18, 1))
    l4 = tf.reduce_mean(tf.abs(p_max_pool4 - gt_max_pool4))
    p_max_pool3=maxpool2d(pred,name='pred_max_pool_3',kshape=(1,shape[2]//8,shape[2]//8,1),strides=(1,shape[2]//8,shape[2]//8,1))
    gt_max_pool3=maxpool2d(gt,name='gt_max_pool_3',kshape=(1,shape[2]//8,shape[2]//8,1),strides=(1,shape[2]//8,shape[2]//8,1))
    l3=tf.reduce_mean(tf.abs(p_max_pool3-gt_max_pool3))
    # shape=p_max_pool3.shape.as_list()

    p_max_pool2=maxpool2d(pred,name='pred_max_pool_2',kshape=(1,shape[2]//4,shape[2]//4,1),strides=(1,shape[2]//4,shape[2]//4,1))
    gt_max_pool2=maxpool2d(gt,name='gt_max_pool_2',kshape=(1,shape[2]//4,shape[2]//4,1),strides=(1,shape[2]//4,shape[2]//4,1))
    l2=tf.reduce_mean(tf.abs(p_max_pool2-gt_max_pool2))
    # shape = p_max_pool2.shape.as_list()
    p_max_pool1=maxpool2d(pred,name='pred_max_pool_1',kshape=(1,shape[2]//2,shape[2]//2,1),strides=(1,shape[2]//2,shape[2]//2,1))
    gt_max_pool1=maxpool2d(gt,name='gt_max_pool_1',kshape=(1,shape[2]//2,shape[2]//2,1),strides=(1,shape[2]//2,shape[2]//2,1))
    l1=tf.reduce_mean(tf.abs(p_max_pool1-gt_max_pool1))
    return 0.5*l1+0.5*l2+l3+l4+l5

# def max_pool_loss(pred,gt):
#     shape=pred.shape.as_list()
#     pred=tf.reshape(pred,[shape[0]*shape[1],shape[2],shape[3],shape[4]])
#     gt=tf.reshape(gt,[shape[0]*shape[1],shape[2],shape[3],shape[4]])
#     p_max_pool5 = maxpool2d(pred, name='pred_max_pool_5', kshape=(1, shape[2] // 36, shape[2] // 36, 1),
#                             strides=(1, shape[2] // 36, shape[2] // 36, 1))
#     gt_max_pool5 = maxpool2d(gt, name='gt_max_pool_5', kshape=(1, shape[2] // 36, shape[2] // 36, 1),
#                              strides=(1, shape[2] // 36, shape[2] // 36, 1))
#     l5 = tf.reduce_mean(tf.abs(p_max_pool5 - gt_max_pool5))
#     p_max_pool4 = maxpool2d(pred, name='pred_max_pool_4', kshape=(1, shape[2] // 18, shape[2] // 18, 1),
#                             strides=(1, shape[2] // 18, shape[2] // 18, 1))
#     gt_max_pool4 = maxpool2d(gt, name='gt_max_pool_4', kshape=(1, shape[2] // 18, shape[2] // 18, 1),
#                              strides=(1, shape[2] // 18, shape[2] // 18, 1))
#     l4 = tf.reduce_mean(tf.abs(p_max_pool4 - gt_max_pool4))
#     p_max_pool3=maxpool2d(pred,name='pred_max_pool_3',kshape=(1,shape[2]//8,shape[2]//8,1),strides=(1,shape[2]//8,shape[2]//8,1))
#     gt_max_pool3=maxpool2d(gt,name='gt_max_pool_3',kshape=(1,shape[2]//8,shape[2]//8,1),strides=(1,shape[2]//8,shape[2]//8,1))
#     l3=tf.reduce_mean(tf.abs(p_max_pool3-gt_max_pool3))
#     # shape=p_max_pool3.shape.as_list()
#
#     p_max_pool2=maxpool2d(pred,name='pred_max_pool_2',kshape=(1,shape[2]//4,shape[2]//4,1),strides=(1,shape[2]//4,shape[2]//4,1))
#     gt_max_pool2=maxpool2d(gt,name='gt_max_pool_2',kshape=(1,shape[2]//4,shape[2]//4,1),strides=(1,shape[2]//4,shape[2]//4,1))
#     l2=tf.reduce_mean(tf.abs(p_max_pool2-gt_max_pool2))
#     # shape = p_max_pool2.shape.as_list()
#     p_max_pool1=maxpool2d(pred,name='pred_max_pool_1',kshape=(1,shape[2]//2,shape[2]//2,1),strides=(1,shape[2]//2,shape[2]//2,1))
#     gt_max_pool1=maxpool2d(gt,name='gt_max_pool_1',kshape=(1,shape[2]//2,shape[2]//2,1),strides=(1,shape[2]//2,shape[2]//2,1))
#     l1=tf.reduce_mean(tf.abs(p_max_pool1-gt_max_pool1))
#     return 0.5*l1+0.5*l2+l3+l4+l5
def layer_norm_compute_python(x, epsilon, scale, bias):
    mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias


def inception_conv_2d(input, kernels, biases, strides, name,padding="SAME"):
    with tf.variable_scope(name):
        result = []
        i=1
        for kernel, bias in zip(kernels, biases):
            out = conv2d_act(input, name+'_incp{}_'.format(i),kernel, bias, 1,padding=padding)
            result.append(out)
            i+=1
        result = tf.concat(result, axis=-1)
    return result
def layer_norm(x, filters=None, epsilon=1e-6, scope=None, reuse=None):
    if filters is None:
        filters = x.get_shape()[-1]
    with tf.variable_scope(scope, default_name="layer_norm", values=[x], reuse=reuse):
        scale = tf.get_variable(
            "layer_norm_scale", [filters], regularizer=None, initializer=tf.ones_initializer())
        bias = tf.get_variable(
            "layer_norm_bias", [filters], regularizer=None, initializer=tf.zeros_initializer())
        result = layer_norm_compute_python(x, epsilon, scale, bias)
        return result

def multi_category_focal_loss1(y_true, y_pred):

    epsilon = 1.e-7
    gamma = 2.0
    # alpha = tf.constant([[2],[1],[1],[1],[1]], dtype=tf.float32)
    alpha = tf.constant([[1], [1], [1], [1], [1], [1], [1], [1], [2], [2], [3], [3], [4], [4], [1], [1], [1]],
                        dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
    ce = -tf.log(y_t)
    weight = tf.pow(tf.subtract(1., y_t), gamma)
    fl = tf.matmul(tf.multiply(weight, ce), alpha)
    loss = tf.reduce_mean(fl)
    return loss

if __name__ == '__main__':

    pred=tf.ones([2,10,360,360,1])
    gt=tf.ones([2,10,360,360,1])
    print(max_pool_loss(pred,gt))
    # A=tf.constant([[1,1,1,1,1,]])
    # alpha = tf.constant([[1], [1], [1], [1], [1], [1], [1], [1], [2], [2], [3], [3], [4], [4], [1], [1], [1]],
    #                     dtype=tf.float32)
    #
    #
    # Y_true = tf.ones(shape=[2*5*700*900,17])
    # Y_pred = tf.ones(shape=[2*5*700*900,17])
    # foval=multi_category_focal_loss1(Y_true, Y_pred)
    # sess=tf.Session()
    # print(sess.run(fetches=foval))

    # gt = np.random.rand(5,5) * 255
    # gt = gt.astype(np.uint8)
    # print(gt)
    # import PIL.Image as IMG
    # gt_img=np.array(IMG.open('/extend/gru_tf_data/gru_experiment/mae/Valid/149999valid/1/out/19.png'),dtype=np.int32)
    # pred_img=np.array(IMG.open('/extend/gru_tf_data/gru_experiment/mae/Valid/149999valid/1/pred/19.png'),dtype=np.int32)
    # hl_20=hing_loss(gt_img,20)
    # a_s1=np.sum(hl_20)
    # hl_30=hing_loss(gt_img, 30)
    # a_s2=np.sum(hl_30)
    # hl_40=hing_loss(gt_img, 40)
    # a_s3=np.sum(hl_40)
    # a_s4=np.sum(hing_loss(gt_img,50))
    # b_s1=np.sum(hing_loss(pred_img,20))
    # b_s2=np.sum(hing_loss(pred_img,30))
    # b_s3=np.sum(hing_loss(pred_img,40))
    # b_s4=np.sum(hing_loss(pred_img,50))
    # print(a_s1,a_s2,a_s3)
    # print(b_s1,b_s2,b_s3)
    # print(a_s1-a_s2,a_s2-a_s3,a_s3-a_s4)
    # print(b_s1-b_s2,b_s2-b_s3,b_s3-b_s4)
    #
    # print('20-30',len(gt_img[gt_img>=20])-len(gt_img[gt_img>=30]))
    # print('30-40',len(gt_img[gt_img>=30])-len(gt_img[gt_img>=40]))
    # print('40-50',len(gt_img[gt_img>=40])-len(gt_img[gt_img>=50]))

    # d_s=np.sum(hing_loss(pred_img,30))
    # print(a_s1)
    # print(b_s1)
    # from iterator import Iterator
    # ite=Iterator(2)
    # data=ite.get_batch()
    # print(hing_loss(data))
    A=tf.zeros([2,360,360,12])
    print(LN(A))
    print(layer_norm(A))