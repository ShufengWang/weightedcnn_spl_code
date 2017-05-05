# -*- coding: utf-8 -*-
import lasagne

from lasagne.layers import Conv2DLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as MaxPoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as LRN2DLayer
from lasagne.layers import DropoutLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer, SliceLayer, concat
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import batch_norm
import theano.tensor as T
from theano.tensor.nnet import logsoftmax
#def log_softmax(x):
#    xdev = x-x.max(1,keepdims=True)
#    return xdev - T.log(T.sum(T.exp(xdev),axis=1,keepdims=True))
#
#
def build_alexnet(cls_num, input_var=None, c=(3, 227, 227)):
    #building the network
    net = {}
    net['data'] = InputLayer(shape=(None, c[0], c[1], c[2]), input_var=input_var)
    # conv1
    net['conv1'] = ConvLayer(
            net['data'],
            num_filters=96,
            filter_size=(11,11),
            stride=4,
            nonlinearity=rectify)
    # pool1
    net['pool1'] = MaxPoolLayer(net['conv1'], pool_size=(3,3), stride=2)
    # norm1
    net['norm1'] = LRN2DLayer(net['pool1'], n=5, alpha=0.0001, beta=0.75, k=1)

    # conv2 use a parameter called group.
    # before conv2 split data
    net['conv2_data1'] = SliceLayer(net['norm1'], indices=slice(0, 48), axis=1)
    net['conv2_data2'] = SliceLayer(net['norm1'], indices=slice(48, 96), axis=1)

    net['conv2_part1'] = Conv2DLayer(net['conv2_data1'],
                                    num_filters=128,
                                    filter_size=(5,5),
                                    pad=2)
    net['conv2_part2'] = Conv2DLayer(net['conv2_data2'],
                                    num_filters=128,
                                    filter_size=(5,5),
                                    pad=2)
    # now conbine
    net['conv2'] = concat((net['conv2_part1'],net['conv2_part2']),axis=1)
    net['pool2'] = MaxPoolLayer(net['conv2'], pool_size=(3, 3), stride = 2)
    net['norm2'] = LRN2DLayer(net['pool2'], n=5, alpha=0.001, beta=0.75, k=1)

    # conv3
    net['conv3'] = Conv2DLayer(
            net['norm2'],
            num_filters=384,
            filter_size=(3,3),
            pad=1)
    net['conv4_data1'] = SliceLayer(net['conv3'], indices=slice(0, 192), axis=1)
    net['conv4_data2'] = SliceLayer(net['conv3'], indices=slice(192,384), axis=1)
    net['conv4_part1'] = Conv2DLayer(net['conv4_data1'],
                                    num_filters=192,
                                    filter_size=(3,3),
                                    pad=1)
    net['conv4_part2'] = Conv2DLayer(net['conv4_data2'],
                                    num_filters=192,
                                    filter_size=(3,3),
                                    pad=1)
    net['conv4'] = concat((net['conv4_part1'],net['conv4_part2']),axis=1)

    # conv5

    net['conv5_data1'] = SliceLayer(net['conv4'], indices=slice(0, 192), axis=1)
    net['conv5_data2'] = SliceLayer(net['conv4'], indices=slice(192,384), axis=1)
    net['conv5_part1'] = Conv2DLayer(net['conv5_data1'],
                                    num_filters=128,
                                    filter_size=(3,3),
                                    pad=1)
    net['conv5_part2'] = Conv2DLayer(net['conv5_data2'],
                                    num_filters=128,
                                    filter_size=(3,3),
                                    pad=1)
    net['conv5'] = concat((net['conv5_part1'],net['conv5_part2']),axis=1)

    # pool 5
    net['pool5'] = MaxPoolLayer(net['conv5'], pool_size=(3, 3), stride = 2)
    # fc6
    net['fc6'] = DenseLayer(
                net['pool5'], num_units=4096,
                nonlinearity=lasagne.nonlinearities.rectify)
    # fc7
    net['fc7'] = DenseLayer(
                net['fc6'],num_units=4096,
                nonlinearity=lasagne.nonlinearities.rectify)

    # fc8
    net['fc8'] = DenseLayer(
                net['fc7'],
                num_units=cls_num,
                nonlinearity=lasagne.nonlinearities.softmax)

    return net

def build_cnn(cls_num, input_var=None, n=5, c=(3, 32, 32)):
    def residual_block(l, increase_dim=False, projection=False):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2,2)
            out_num_filters = input_num_filters * 2
        else:
            first_stride = (1,1)
            out_num_filters = input_num_filters

        stack_1 = batch_norm(ConvLayer(l, num_filters=out_num_filters,
                                       filter_size=(3,3),
                                       stride=first_stride,
                                       nonlinearity=rectify,
                                       pad='same',
                                       W=lasagne.init.HeNormal(gain='relu'),
                                       flip_filters=False))
        stack_2 = batch_norm(ConvLayer(stack_1,
                                       num_filters=out_num_filters,
                                       filter_size=(3,3),
                                        stride=(1,1),
                                        nonlinearity=None,
                                        pad='same',
                                        W=lasagne.init.HeNormal(gain='relu'),
                                        flip_filters=False))

        # add shortcut connections
        if increase_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = batch_norm(ConvLayer(l,
                                                  num_filters=out_num_filters,
                                                  filter_size=(1,1),
                                                  stride=(2,2),
                                                  nonlinearity=None,
                                                  pad='same', b=None,
                                                  flip_filters=False))
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]),nonlinearity=rectify)
            else:
                # identity shortcut, as option A in paper
                identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
                padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]),nonlinearity=rectify)
        else:
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]),nonlinearity=rectify)

        return block

    # Building the network
    l_in = InputLayer(shape=(None, c[0], c[1], c[2]), input_var=input_var)

    # first layer, output is 16 x 32 x 32
    l = batch_norm(ConvLayer(l_in, num_filters=16,
                             filter_size=(3,3), stride=(2,2),
                             nonlinearity=rectify, pad='same',
                             W=lasagne.init.HeNormal(gain='relu'),
                             flip_filters=False))

    # first stack of residual blocks, output is 16 x 32 x 32
    for _ in range(n):
        l = residual_block(l)

    # second stack of residual blocks, output is 32 x 16 x 16
    l = residual_block(l, increase_dim=True)
    for _ in range(n):
        l = residual_block(l)

    l = residual_block(l, increase_dim=True)
    l1 = l2 = l
    for _ in range(1,n):
        l1 = residual_block(l1)

    # third stack of residual blocks, output is 64 x 8 x 8
    l1 = residual_block(l1, increase_dim=True)

    for _ in range(1,n):
        l1 = residual_block(l1)

    # average pooling
    l1 = GlobalPoolLayer(l1)

    # fully connected layer
    cls_network = DenseLayer(
            l1, num_units=cls_num,
            W=lasagne.init.HeNormal(),
            nonlinearity=softmax)

    ###############kl line##################
    for _ in range(1,n):
        l2 = residual_block(l2)

    # third stack of residual blocks, output is 64 x 8 x 8
    l2 = residual_block(l2, increase_dim=True)
    for _ in range(1,n):
        l2 = residual_block(l2)

    # average pooling
    l2 = GlobalPoolLayer(l2)

    # fully connected layer
    kl_network = DenseLayer(
            l2, num_units=cls_num,
            W=lasagne.init.HeNormal(),
            nonlinearity=softmax)

    return cls_network, kl_network
