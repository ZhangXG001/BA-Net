
import os
import tensorflow as tf
import numpy as np
from network import*
from math import ceil
from ops import res_block
from tflearn.layers.conv import global_avg_pool
num_class = 2  
h = 300 
w = 400 
batch_size = 5

def pool2d(x, pool_size, pool_stride, name):
    pool = tf.layers.max_pooling2d(x, pool_size, pool_stride, name='pool_{}'.format(name), padding='same')
    return pool


def get_deconv_filter(filter_shape, upscale_factor):
    ##filter_shape is [width, height, num_in_channels, num_out_channels]
    kernel_size = filter_shape[1]
    ### Centre location of the filter for which value is calculated
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            ##Interpolation Calculation
            value = (1 - abs((x - centre_location) / upscale_factor)) * (
                1 - abs((y - centre_location) / upscale_factor))
            bilinear[x, y] = value
    weights = np.zeros(filter_shape)
    for i in range(filter_shape[2]):
        weights[:, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)

    bilinear_weights = tf.get_variable(initializer=init, name="up_filter",
                                       shape=weights.shape, trainable=False)  # ,,trainable=False
    return bilinear_weights

def deconv2d(inputT, f_shape, output_shape, stride=2, name=None):
  # output_shape = [b, w, h, c]
  # sess_temp = tf.InteractiveSession()
  sess_temp = tf.global_variables_initializer()
  strides = [1, stride, stride, 1]
  with tf.variable_scope(name):
    weights = get_deconv_filter(f_shape,stride)
    deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape,
                                        strides=strides, padding='SAME')
  return deconv


def batch_norm_layer(inputT, is_training, scope):
  return tf.cond(is_training,
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                           center=False, updates_collections=None, scope=scope+"_bn"),
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                          updates_collections=None, center=False, scope=scope+"_bn", reuse = True))

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x) :
    return tf.nn.sigmoid(x)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Fully_connected(x, units, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
         return tf.layers.dense(inputs=x, use_bias=False, units=units)

def Squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name) :
        squeeze = Global_Average_Pooling(input_x)       
        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')        
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
        excitation = Sigmoid(excitation) 
        excitation = tf.reshape(excitation, [-1,1,1,out_dim])
        scale = input_x * excitation
        return scale
    

def unet(input, training):

    image_R = input[:,:,:, 0:1]    
    image_G = input[:,:,:, 1:2] 
    image_B = input[:,:,:, 2:3]
    max_R=tf.reduce_max(image_R)
    min_R=tf.reduce_min(image_R)
    max_G=tf.reduce_max(image_G)
    min_G=tf.reduce_min(image_G)
    max_B=tf.reduce_max(image_B)
    min_B=tf.reduce_min(image_B)
    input_1 = 2*(image_R-min_R)/(max_R-min_R)-1
    input_2 = 2*(image_G-min_G)/(max_G-min_G)-1
    input_3 = 2*(image_B-min_B)/(max_B-min_B)-1
    
   
    input = tf.concat([input_1, input_2,input_3], axis=-1,name='inputs')
    side_layer1, side_layer2, side_layer3, side_layer4, side_layer5 = network(input,training)
    #################################################################################################################################
  
    side_layer1_1 = conv_3x3(side_layer1, output_dim = 16, is_train = training, name = '1_1', bias=False)    
    side_layer2_1 = conv_3x3(side_layer2, output_dim = 16, is_train = training, name = '1_2', bias=False)
    side_layer3_1 = conv_3x3(side_layer3, output_dim = 16, is_train = training, name = '1_3', bias=False)
    side_layer4_1 = conv_3x3(side_layer4, output_dim = 16, is_train = training, name = '1_4', bias=False)
    side_layer5_1 = conv_3x3(side_layer5, output_dim = 16, is_train = training, name = '1_5', bias=False)

    side_layer5_2 = conv_1x1(side_layer5_1, output_dim=1, name='side_layer5_2', bias=False)
    #################################################################################################################################
    side_layer5_1_to_4_2 = deconv2d(side_layer5_1, f_shape=[4,4,16,16], output_shape=[batch_size,19, 25,16], stride=2, name='side_layer5_1_to_4_2')
    side_layer4_1_to_4_2 = side_layer4_1
    side_layer4_2_concat = tf.concat([side_layer5_1_to_4_2, side_layer4_1_to_4_2], axis=-1, name='side_layer4_2_concat')
    #side_layer4_2_concat = side_layer_conv(side_layer4_2_concat, exp=4, output_dim=8, is_train=training, name='side_layer4_2',reuse=False)
    SE_4_2 = Squeeze_excitation_layer(side_layer4_2_concat, out_dim =side_layer4_2_concat.get_shape()[-1].value, ratio=2, layer_name = 'SE_4_2')
    
    side_layer4_2 = conv_3x3(SE_4_2, output_dim = 16, is_train = training, name = 's4_2', bias=False)
    
    side_layer4_3 = conv_1x1(side_layer4_2, output_dim=1, name='side_layer4_3', bias=False)

    

    side_layer4_1_to_3_2 = deconv2d(side_layer4_1, f_shape=[4,4,16,16], output_shape=[batch_size,38, 50,16], stride=2, name='side_layer4_1_to_3_2')
    side_layer3_1_to_3_2 = side_layer3_1
    side_layer3_2_concat = tf.concat([side_layer4_1_to_3_2,side_layer3_1_to_3_2], axis=-1, name='side_layer3_2_concat')
    #side_layer3_2_concat = side_layer_conv(side_layer3_2_concat, exp=4, output_dim=8, is_train=training, name='side_layer3_2',reuse=False)
    SE_3_2 = Squeeze_excitation_layer(side_layer3_2_concat, out_dim =side_layer3_2_concat.get_shape()[-1].value, ratio=2, layer_name = 'SE_3_2')
    
    side_layer3_2 = conv_3x3(SE_3_2, output_dim = 16, is_train = training, name = 's3_2', bias=False)



    side_layer3_1_to_2_2 = deconv2d(side_layer3_1, f_shape=[4,4,16,16], output_shape=[batch_size,75, 100,16], stride=2, name='side_layer3_1_to_2_2')
    side_layer2_1_to_2_2 = side_layer2_1
    side_layer2_2_concat = tf.concat([side_layer3_1_to_2_2, side_layer2_1_to_2_2], axis=-1,name='side_layer2_2_concat')
    #side_layer2_2_concat = side_layer_conv(side_layer2_2_concat, exp=4, output_dim=8, is_train=training, name='side_layer2_2',reuse=False)
    SE_2_2 = Squeeze_excitation_layer(side_layer2_2_concat, out_dim =side_layer2_2_concat.get_shape()[-1].value, ratio=2, layer_name = 'SE_2_2')
    
    side_layer2_2 = conv_3x3(SE_2_2, output_dim = 16, is_train = training, name = 's2_2', bias=False)



    side_layer2_1_to_1_2 = deconv2d(side_layer2_1, f_shape=[4,4,16,16], output_shape=[batch_size,150, 200,16], stride=2, name='side_layer2_1_to_1_2')
    side_layer1_1_to_1_2 = side_layer1_1
    side_layer1_2_concat = tf.concat( [side_layer2_1_to_1_2,side_layer1_1_to_1_2], axis=-1,name='side_layer1_2_concat')
    #side_layer1_2_concat = side_layer_conv(side_layer1_2_concat, exp=4, output_dim=8, is_train=training, name='side_layer1_2',reuse=False)
    SE_1_2 = Squeeze_excitation_layer(side_layer1_2_concat, out_dim =side_layer1_2_concat.get_shape()[-1].value, ratio=2, layer_name = 'SE_1_2')
    
    side_layer1_2 = conv_3x3(SE_1_2, output_dim = 16, is_train = training, name = 's1_2', bias=False)
    
    #################################################################################################################################
    side_layer4_2_to_3_3 = deconv2d(side_layer4_2, f_shape=[4,4,16,16], output_shape=[batch_size,38, 50,16], stride=2, name='side_layer4_2_to_3_3')
    side_layer3_2_to_3_3 = side_layer3_2
    side_layer3_3_concat = tf.concat([side_layer4_2_to_3_3, side_layer3_2_to_3_3], axis=-1,name='side_layer3_3_concat')
    #side_layer3_3_concat = side_layer_conv(side_layer3_3_concat, exp=4, output_dim=8, is_train=training, name='side_layer3_3',reuse=False)
    SE_3_3 = Squeeze_excitation_layer(side_layer3_3_concat, out_dim =side_layer3_3_concat.get_shape()[-1].value, ratio=2, layer_name = 'SE_3_3')
    
    side_layer3_3 = conv_3x3(SE_3_3, output_dim = 16, is_train = training, name = 's3_3', bias=False)

    side_layer3_4 = conv_1x1(side_layer3_3, output_dim=1, name='side_layer3_4', bias=False)

   
    side_layer3_2_to_2_3 = deconv2d(side_layer3_2, f_shape=[4,4,16,16], output_shape=[batch_size,75, 100,16], stride=2, name='side_layer3_2_to_2_3')
    side_layer2_2_to_2_3 = side_layer2_2
    side_layer2_3_concat = tf.concat([side_layer3_2_to_2_3, side_layer2_2_to_2_3,], axis=-1,name='side_layer2_3_concat')
    #side_layer2_3_concat = side_layer_conv(side_layer2_3_concat, exp=4, output_dim=8, is_train=training, name='side_layer2_3',reuse=False)
    SE_2_3 = Squeeze_excitation_layer(side_layer2_3_concat, out_dim =side_layer2_3_concat.get_shape()[-1].value, ratio=2, layer_name = 'SE_2_3')
    
    side_layer2_3 = conv_3x3(SE_2_3, output_dim = 16, is_train = training, name = 's2_3', bias=False)

    side_layer2_2_to_1_3 = deconv2d(side_layer2_2, f_shape=[4,4,16,16], output_shape=[batch_size,150, 200,16], stride=2, name='side_layer2_2_to_1_3')
    side_layer1_2_to_1_3 = side_layer1_2
    side_layer1_3_concat = tf.concat([side_layer2_2_to_1_3, side_layer1_2_to_1_3], axis=-1,name='side_layer1_3_concat')
    #side_layer1_3_concat = side_layer_conv(side_layer1_3_concat, exp=4, output_dim=8, is_train=training, name='side_layer1_3',reuse=False)
    SE_1_3 = Squeeze_excitation_layer(side_layer1_3_concat, out_dim =side_layer1_3_concat.get_shape()[-1].value, ratio=2, layer_name = 'SE_1_3')
    
    side_layer1_3 = conv_3x3(SE_1_3, output_dim = 16, is_train = training, name = 's1_3', bias=False)
    #################################################################################################################################
    side_layer3_3_to_2_4 = deconv2d(side_layer3_3, f_shape=[4,4,16,16], output_shape=[batch_size,75, 100,16], stride=2, name='side_layer3_3_to_2_4')
    side_layer2_3_to_2_4 = side_layer2_3
    side_layer2_4_concat = tf.concat([side_layer3_3_to_2_4, side_layer2_3_to_2_4], axis=-1, name='side_layer2_4_concat')
    #side_layer2_4_concat = side_layer_conv(side_layer2_4_concat, exp=4, output_dim=8, is_train=training, name='side_layer2_4',reuse=False)
    SE_2_4 = Squeeze_excitation_layer(side_layer2_4_concat, out_dim =side_layer2_4_concat.get_shape()[-1].value, ratio=2, layer_name = 'SE_2_4')
    
    side_layer2_4 = conv_3x3(SE_2_4, output_dim = 16, is_train = training, name = 's2_4', bias=False)
    side_layer2_5 = conv_1x1(side_layer2_4, output_dim=1, name='side_layer2_5', bias=False)


    side_layer2_3_to_1_4 = deconv2d(side_layer2_3, f_shape=[4,4,16,16], output_shape=[batch_size,150, 200,16], stride=2, name='side_layer2_3_to_1_4')
    side_layer1_3_to_1_4 = side_layer1_3
    side_layer1_4_concat = tf.concat([side_layer2_3_to_1_4, side_layer1_3_to_1_4], axis=-1,name='side_layer1_4_concat')
    #side_layer1_4_concat = side_layer_conv(side_layer1_4_concat, exp=4, output_dim=8, is_train=training, name='side_layer1_4',reuse=False)
    SE_1_4 = Squeeze_excitation_layer(side_layer1_4_concat, out_dim =side_layer1_4_concat.get_shape()[-1].value, ratio=2, layer_name = 'SE_1_4')
    
    side_layer1_4 = conv_3x3(SE_1_4, output_dim = 16, is_train = training, name = 's1_4', bias=False)
    #################################################################################################################################
    side_layer2_4_to_1_5 = deconv2d(side_layer2_4, f_shape=[4,4,16,16], output_shape=[batch_size,150, 200,16], stride=2, name='side_layer2_4_to_1_5')
    side_layer1_4_to_1_5 = side_layer1_4
    side_layer1_5_concat = tf.concat([side_layer2_4_to_1_5,side_layer1_4_to_1_5], axis=-1,name='side_layer1_5_concat')
    #side_layer1_5_concat = side_layer_conv(side_layer1_5_concat, exp=4, output_dim=8, is_train=training, name='side_layer1_5',reuse=False)
    SE_1_5 = Squeeze_excitation_layer(side_layer1_5_concat, out_dim =side_layer1_5_concat.get_shape()[-1].value, ratio=2, layer_name = 'SE_1_5')
    
    side_layer1_5 = conv_3x3(SE_1_5, output_dim = 16, is_train = training, name = 's1_5', bias=False)

    side_layer1_6 = conv_1x1(side_layer1_5, output_dim=1, name='side_layer1_6', bias=False)
    
    #################################################################################################################################  
    side_layer1 = deconv2d(side_layer1_6, f_shape=[4,4,1,1], output_shape=[batch_size,h, w,1], stride=2, name='side_layer1')
    side_layer2 = deconv2d(side_layer2_5, f_shape=[8,8,1,1], output_shape=[batch_size,h, w,1], stride=4, name='side_layer2')
    side_layer3 = deconv2d(side_layer3_4, f_shape=[16,16,1,1], output_shape=[batch_size,h, w,1], stride=8, name='side_layer3')
    side_layer4 = deconv2d(side_layer4_3, f_shape=[32,32,1,1], output_shape=[batch_size,h, w,1], stride=16, name='side_layer4')
    side_layer5 = deconv2d(side_layer5_2, f_shape=[64,64,1,1], output_shape=[batch_size,h, w,1], stride=32, name='side_layer5')

    concat_upscore = tf.concat([side_layer1, side_layer2, side_layer3,side_layer4,side_layer5],axis=-1, name='concat_upscore')
    upscore_fuse = tf.layers.conv2d(concat_upscore, filters=1, kernel_size=(1, 1), strides=(1, 1), padding='SAME',
                                    name='output1', kernel_initializer=tf.constant_initializer(
                                    0.2)) 
    #upscore_fuse =0.2*side_layer1+0.2*side_layer2+0.2*side_layer3+0.2*side_layer4+0.2*side_layer5
    return  side_layer5, side_layer4,side_layer3,side_layer2, side_layer1, upscore_fuse

def loss_CE(y_pred, y_true):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred) 
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    return cross_entropy_mean
