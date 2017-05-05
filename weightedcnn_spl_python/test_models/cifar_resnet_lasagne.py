# -*- coding: utf-8 -*-
#3 cifar10_deep_residual_model.npz
from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import lasagne
theano.config.exception_verbosity='low'


# for the larger networks (n>=9), we need to adjust pythons recursion limit
sys.setrecursionlimit(10000)

import sys
sys.path.append('/media/dl/data2/cifar10_resnet/')  

from utils import *
from dec_model import *
# ############################## Main program ################################

def main(data, cls_num, increase=True, n=2, num_epochs=500, model=None):
    # Check if cifar data exists
    if not os.path.exists("/media/dl/data1/cifar10/cifar-10-batches-py/"):
        print("CIFAR-10 dataset can not be found. Please download the dataset from 'https://www.cs.toronto.edu/~kriz/cifar.html'.")
        return
    X_train = data['X_train']
    Y_train = data['Y_train']    
    X_val = data['X_val']
    Y_val = data['Y_val']
    X_test = data['X_test']
    Y_test = data['Y_test']
    
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    batchsize = 128
    
    # Create neural network model
    print("Building model and compiling functions...")
    network, l = build_cnn(cls_num, input_var, increase, n)
    print("number of parameters in model: %d" % lasagne.layers.count_params(network, trainable=True))
    print(architecture_string(network))
    if model is None:
        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        prediction = lasagne.layers.get_output(network)
        
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        # add weight decay
        all_layers = lasagne.layers.get_all_layers(network)
        l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
        loss = loss + l2_penalty
        
        lr = 0.1
        sh_lr = theano.shared(lasagne.utils.floatX(lr))
        params = lasagne.layers.get_all_params(network, trainable=True)
#        all_grads = T.grad(loss, params)
#        scaled_grads = lasagne.updates.total_norm_constraint(all_grads, 10)
        updates = lasagne.updates.momentum(
                loss, params, learning_rate=sh_lr, momentum=0.9)
        #updates = lasagne.updates.adam(loss, params)
        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var), dtype=theano.config.floatX)
        train_fn = theano.function([input_var, target_var], [loss, train_acc], updates=updates)
    # Create a loss expression for validation/testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc, test_prediction])
    
    if model is None:
        # launch the training loop
        print("Starting training...")
        # We iterate over epochs:
        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            train_acc = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, Y_train, batchsize, shuffle=True, augment=True):
                inputs, targets = batch
                err, acc = train_fn(inputs, targets)
#                cls0_feat_val, cls1_feat_val, margin_loss_val = margin_loss_fun(inputs, targets)
#                print('margin loss %d' % margin_loss_val)
#                print(cls0_feat_val, cls1_feat_val)
                train_err += err
                train_acc += acc
                train_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  training accuracy:\t\t{:.2f} %".format(
                train_acc / train_batches * 100))
            
            getpreds(X_val, Y_val, val_fn, 'val', batchsize)
            if (epoch+1) == 41 or (epoch+1) == 61:
                new_lr = sh_lr.get_value() * 0.1
                print("New LR:"+str(new_lr))
                sh_lr.set_value(lasagne.utils.floatX(new_lr))
            
        # dump the network weights to a file :
        np.savez('cifar10.npz', *lasagne.layers.get_all_param_values(network))
    else:
        # load network weights from model file
        with np.load(model) as f:
             param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)
    getpreds(X_test, Y_test, val_fn, 'val', batchsize)
    
if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a Deep Residual Learning network on cifar-10 using Lasagne.")
        print("Network architecture and training parameters are as in section 4.2 in 'Deep Residual Learning for Image Recognition'.")
        print("Usage: %s [N [MODEL]]" % sys.argv[0])
        print()
        print("N: Number of stacked residual building blocks per feature map (default: 5)")
        print("MODEL: saved model file to load (for validation) (default: None)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['n'] = int(sys.argv[1])
        else:
            kwargs['n'] = 5
        if len(sys.argv) > 2:
            kwargs['model'] = sys.argv[2]
        # Load the dataset
        print("Loading data...")
        data = load_data(10)
        data = split_data(data, 45000, 5000)
        
        kwargs['cls_num'] = 10
        kwargs['num_epochs'] = 200
        kwargs['increase'] = True
        main(data, **kwargs)
        
        
        