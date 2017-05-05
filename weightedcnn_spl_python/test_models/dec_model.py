# -*- coding: utf-8 -*-
import lasagne

from lasagne.layers import Conv2DLayer as ConvLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import batch_norm
from lasagne.layers import DropoutLayer
def build_cnn(cls_num, input_var=None, increase=True, n=5, c=(3, 32, 32)):
    if increase:
        init_filters = 16
    else:
        init_filters = 64
    def residual_block(l, increase_dim=False, projection=False, dropout=0):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2,2)
            if increase:
                out_num_filters = input_num_filters * 2
            else:
                out_num_filters = input_num_filters / 2
        else:
            first_stride = (1,1)
            out_num_filters = input_num_filters
        
        stack_1 = batch_norm(DropoutLayer(
                            ConvLayer(l, num_filters=out_num_filters, 
                                       filter_size=(3,3),
                                       stride=first_stride,
                                       nonlinearity=rectify,
                                       pad='same',
                                       W=lasagne.init.HeNormal(gain='relu'), 
                                       flip_filters=False), p = dropout))
        stack_2 = batch_norm(DropoutLayer(ConvLayer(stack_1, 
                                       num_filters=out_num_filters, 
                                       filter_size=(3,3),
                                        stride=(1,1), 
                                        nonlinearity=None, 
                                        pad='same',
                                        W=lasagne.init.HeNormal(gain='relu'),
                                        flip_filters=False), p = dropout))                      
    
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
    l = batch_norm(ConvLayer(l_in, num_filters=init_filters, 
                             filter_size=(3,3), stride=(1,1), 
                             nonlinearity=rectify, pad='same', 
                             W=lasagne.init.HeNormal(gain='relu'), 
                             flip_filters=False))
    
    # first stack of residual blocks, output is 16 x 32 x 32
    for _ in range(n):
        l = residual_block(l)

    # second stack of residual blocks, output is 32 x 16 x 16
    l = residual_block(l, increase_dim=True, projection=True)
    for _ in range(1,n):
        l = residual_block(l)

    # third stack of residual blocks, output is 64 x 8 x 8
    l = residual_block(l, increase_dim=True, projection=True)
    for _ in range(1,n):
        l = residual_block(l)
    
    # average pooling
    l = GlobalPoolLayer(l)
    #l = DenseLayer(l, num_units=64)

    # fully connected layer
    network = DenseLayer(
            l, num_units=cls_num,
            W=lasagne.init.HeNormal(),
            nonlinearity=softmax)

    return network, l