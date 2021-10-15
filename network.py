import time
from ops import *
#from utils import *

def network(input,training, reuse=False):
    print(input)
    print(training)
    net=conv2d_block(input, 16, 5, 2, is_train=training, name='resblock0')
    sidelayer1=net
    print('sidelayer1',sidelayer1)
    with tf.variable_scope("main_network", reuse=reuse):
        residual_list = get_residual_layer(10)
        exp = 6     
        ########################################################################################################
        net = res_block(net, exp, 16, 2, is_train=training, name='resblock1_0')
        for i in range(1,residual_list[0]):
            net = res_block(net, exp, 16, 1, is_train=training, name='resblock1_' + str(i))
        #sidelayer2 = net
        sidelayer2 = net
        print('sidelayer2',sidelayer2)
        ########################################################################################################
        net = res_block(net, exp, 32, 2, is_train=training, name='resblock2_0')
        for i in range(1,residual_list[1]):
            net = res_block(net, exp, 32, 1, is_train=training, name='resblock2_' + str(i))
        sidelayer3 = net
        print('sidelayer3',sidelayer3)
        ########################################################################################################
        net = res_block(net, exp, 32, 2, is_train=training, name='resblock3_0')
        for i in range(1, residual_list[2]):
            net = res_block(net, exp, 32, 1, is_train=training, name='resblock3_'+ str(i))
        sidelayer4 = net
        print('sidelayer4',sidelayer4)
        ########################################################################################################
        net = res_block(net, exp, 32, 2, is_train=training, name='resblock4_0')
        for i in range(1, residual_list[3]):
            net = res_block(net, exp, 32, 1, is_train=training, name='resblock4_' + str(i))
        sidelayer5 = net
        print('sidelayer5',sidelayer5)
        ########################################################################################################
        return sidelayer1,sidelayer2,sidelayer3,sidelayer4,sidelayer5
def side_layer_conv(net,exp=4,output_dim=8,is_train=True,name='side_conv', reuse=False):
    with tf.variable_scope("name"):
        net = res_block(net, exp, output_dim, stride=1, is_train=is_train, name=name)
        return net
