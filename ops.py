import tensorflow as tf

weight_decay=1e-4

def relu(x, name='relu6'):
    return tf.nn.relu6(x, name)


def batch_norm(x, momentum=0.9, epsilon=1e-5, train=True, name='bn'):
    return tf.layers.batch_normalization(x,
                      momentum=momentum,
                      epsilon=epsilon,
                      scale=True,
                      training=train,
                      name=name)


def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name='conv2d', bias=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        if bias:
            biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv


def conv2d_block(input, out_dim, k, s, is_train, name):
    with tf.name_scope(name), tf.variable_scope(name):
        net = conv2d(input, out_dim, k, k, s, s, name='conv2d_block')
        net = batch_norm(net, train=is_train, name='bn')
        net = relu(net)
        return net


def conv_1x1(input, output_dim, name, bias=False):
    with tf.name_scope(name):
        return conv2d(input, output_dim, 1,1,1,1, stddev=0.02, name=name, bias=bias)

def conv(input, output_dim, name, bias=False):
    with tf.name_scope(name):
        net = conv2d(input, output_dim, 3,3,1,1, stddev=0.02, name=name, bias=bias)
        return net 

def conv_3x3(input, output_dim, is_train, name, bias=False):
    with tf.name_scope(name),tf.variable_scope(name):
        net = conv2d(input, output_dim, 3,3,1,1, stddev=0.02, name='conv_3x3', bias=bias)
        net = batch_norm(net, train=is_train, name='bn')
        net = relu(net)
        return net 

def conv_1x1_2(input, output_dim, name,bias=False):
    with tf.name_scope(name),tf.variable_scope(name):
        net = conv2d(input, output_dim, 1,1,2,2, stddev=0.02, name=name, bias=bias)
        net = batch_norm(net, train=is_train, name='bn')
        net = relu(net)
        return net


def pwise_block(input, output_dim, is_train, name, bias=False):
    with tf.name_scope(name), tf.variable_scope(name):
        out=conv_1x1(input, output_dim, bias=bias, name='pwb')
        out=batch_norm(out, train=is_train, name='bn')
        out=relu(out)
        return out


def dwise_conv(input, k_h=3, k_w=3, channel_multiplier= 1, strides=[1,1,1,1],
               padding='SAME', stddev=0.02, name='dwise_conv', bias=False):
    with tf.variable_scope(name):
        in_channel=input.get_shape().as_list()[-1]
        w = tf.get_variable('w', [k_h, k_w, in_channel, channel_multiplier],
                        regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None,name=None,data_format=None)
        if bias:
            biases = tf.get_variable('bias', [in_channel*channel_multiplier], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv


def res_block(input, expansion_ratio, output_dim, stride, is_train, name, bias=False, shortcut=True):
    with tf.name_scope(name), tf.variable_scope(name):
        # pw
        bottleneck_dim=round(expansion_ratio*input.get_shape().as_list()[-1])
        net = conv_1x1(input, bottleneck_dim, name='pw/1', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_bn')
        net = relu(net)
        # dw
        net = dwise_conv(net, strides=[1, stride, stride, 1], name='dw', bias=bias)
        net = batch_norm(net, train=is_train, name='dw_bn')
        net = relu(net)
        # pw & linear
        net = conv_1x1(net, output_dim, name='pw_linear', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_linear_bn')

        # element wise add, only for stride==1
        if shortcut and stride == 1:
            in_dim=int(input.get_shape().as_list()[-1])
            if in_dim != output_dim:
                ins=conv_1x1(input, output_dim, name='ex_dim')
                net=ins+net
            else:
                net=input+net

        return net


def separable_conv(input, k_size, output_dim, stride, pad='SAME', channel_multiplier=1, name='side_conv', bias=False):
    with tf.name_scope(name), tf.variable_scope(name):
        in_channel = input.get_shape().as_list()[-1]
        dwise_filter = tf.get_variable('dw', [k_size, k_size, in_channel, channel_multiplier],
                  regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                  initializer=tf.truncated_normal_initializer(stddev=0.02))

        pwise_filter = tf.get_variable('pw', [1, 1, in_channel*channel_multiplier, output_dim],
                  regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        strides = [1,stride, stride,1]

        conv=tf.nn.separable_conv2d(input,dwise_filter,pwise_filter,strides,padding=pad, name=name)
        if bias:
            biases = tf.get_variable('bias', [output_dim],initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv


def global_avg(x):
    with tf.name_scope('global_avg'):
        net=tf.layers.average_pooling2d(x, x.get_shape()[1:-1], 1)
        return net


def flatten(x):
    #flattened=tf.reshape(input,[x.get_shape().as_list()[0], -1])  # or, tf.layers.flatten(x)
    return tf.contrib.layers.flatten(x)


def pad2d(inputs, pad=(0, 0), mode='CONSTANT'):
    paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
    net = tf.pad(inputs, paddings, mode=mode)
    return net
def get_residual_layer(res_n) :
    x = []
    if res_n == 10 :
        x = [2,2,2,2]
    return x
def get_side_conv_layer(side_conv_layer) :
    x = []
    if side_conv_layer == 1 :
        x = [1,1,1,1,1]
    if side_conv_layer == 2 :
        x = [1,1,1,1]
    if side_conv_layer == 3 :
        x = [1,1,1]
    if side_conv_layer == 4 :
        x = [1,1]
    if side_conv_layer == 5 :
        x = [1]
    return x
def pool(layer_name, x, ksize=None, stride=None, is_max_pool=True):
    """
    Pooling op
    :param layer_name: layer name, eg:pool1, pool2,...
    :param x:input tensor
    :param ksize:pool kernel size, VGG paper use [1,2,2,1], the size of 2X2
    :param stride:stride size, VGG paper use [1,2,2,1]
    :param is_max_pool: default use max pool, if it is false, the we will use avg_pool
    :return: tensor
    """
    ksize = ksize if ksize else [1, 2, 2, 1]
    stride = stride if stride else [1, 2, 2, 1]

    if is_max_pool:
        x = tf.nn.max_pool(x, ksize, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, ksize, strides=stride, padding='SAME', name=layer_name)

    return x
