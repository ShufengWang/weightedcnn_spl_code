# -*- coding: utf-8 -*-
#3 cifar10_deep_residual_model.npz
from __future__ import print_function
import sys
import os
import time
import string
import random
import pickle

import numpy as np

sys.path.append('../')
# for the larger networks (n>=9), we need to adjust pythons recursion limit
sys.setrecursionlimit(10000)
from utils import get_mtresults, load_data, split_data, architecture_string, printacc, printacc2, getFeat, load_svhn

import theano
import theano.tensor as T
import lasagne

from scipy.spatial.distance import cdist
from cotrain_model import build_cnn
from losses import get_klloss
# ############################## Main program ################################

def main(data, cls_num, increase=True, n=None, num_epochs=500, strategy='momentum', model=None, lb_num=None):


    X_label = data['X_label']
    Y_label = data['Y_label']
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_val = data['X_val']
    Y_val = data['Y_val']
    X_test = data['X_test']
    Y_test = data['Y_test']
    X_unlabel = data['X_unlabel']
    Y_unlabel = data['Y_unlabel']
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    print(Y_train.shape[0])
    # Create neural network model
    print("Building model and compiling functions...")
    cls_network, kl_network = build_cnn(cls_num, input_var, n)

    epsilon = 1e-35
    margin = 2
    batchsize = target_var.shape[0]

    cls_prediction = lasagne.layers.get_output(cls_network)
    kl_prediction = lasagne.layers.get_output(kl_network)

    # add weight decay
    all_layers = lasagne.layers.get_all_layers([cls_network, kl_network])
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0005

    cls_loss = lasagne.objectives.categorical_crossentropy(cls_prediction, target_var)
    cls_loss = cls_loss.mean()

    kl_prediction = kl_prediction + epsilon

    kl_loss = get_klloss(batchsize, margin, kl_prediction, target_var)
    loss = l2_penalty + kl_loss + 0*cls_loss

    lr = 0.1
    sh_lr = theano.shared(lasagne.utils.floatX(lr))
    params = lasagne.layers.get_all_params([cls_network, kl_network], trainable=True)
    if strategy=='momentum':
        updates = lasagne.updates.momentum(loss, params, learning_rate=sh_lr, momentum=0.9)#nesterov_
    else:
        updates = lasagne.updates.adam(loss, params, learning_rate=1e-4)

    train_fn = theano.function([input_var, target_var], [loss, cls_prediction, kl_prediction], updates=updates)
    # Create a loss expression for validation/testing
    test_cls_prediction = lasagne.layers.get_output(cls_network, deterministic=True)
    test_cls_loss = lasagne.objectives.categorical_crossentropy(test_cls_prediction, target_var)
    test_cls_loss = test_cls_loss.mean()

    test_kl_prediction = lasagne.layers.get_output(kl_network, deterministic=True)
    test_kl_prediction = test_kl_prediction + epsilon
    test_kl_loss = get_klloss(batchsize, margin, test_kl_prediction, target_var)

    test_loss = l2_penalty + test_kl_loss + test_cls_loss

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_cls_prediction, test_kl_prediction])

    print("number of parameters in model: %d" % lasagne.layers.count_params([cls_network, kl_network], trainable=True))
    #print(architecture_string([cls_network, kl_network]))
    if model is not None:
        # load network weights from model file
        with np.load(model) as f:
             param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        with np.load('mt_%d_indexes_%d.npz' % (lb_num,n)) as f:
            indexes = f['arr_0']
        lasagne.layers.set_all_param_values([cls_network, kl_network], param_values)
    else:
        # launch the training loop
        print("Starting training...")
        for epoch in np.arange(num_epochs):
            print("Epoch {} of {}:".format(epoch + 1, num_epochs))
            cls_preds, kl_preds, my_hat, indexes, cm = get_mtresults(X_train, Y_train, 'train', train_fn, True, True)
            cls_preds, kl_preds, my_hat, indexes, cm = get_mtresults(X_val, Y_val, 'val', val_fn, indexes = indexes)
            if (epoch+1) == 120 or (epoch+1) == 130 or (epoch+1)==140:
                new_lr = sh_lr.get_value() * 0.1
                print("New LR:"+str(new_lr))
                sh_lr.set_value(lasagne.utils.floatX(new_lr))

        # dump the network weights to a file :
        model_path='model_mt_%d_%d.npz' % (lb_num,n)
        if not os.path.exists(model_path):
            np.savez(model_path, *lasagne.layers.get_all_param_values([cls_network, kl_network]))
            np.savez('mt_%d_indexes_%d.npz' % (lb_num,n), indexes)

    get_mtresults(X_test, Y_test, 'test', val_fn, indexes = indexes)
    return input_var, cls_network, kl_network, val_fn, indexes

def comp_preds(cls_preds, kl_preds, kl_yhat, Y_unlabel):
    unlabel_num = kl_preds.shape[0]
    def getidx(preds, cls):
        plabel = np.argmax(preds, axis=1)
        tpreds = preds[np.arange(unlabel_num), plabel]
        pidx = np.argsort(tpreds)
        print('prediction of ', cls)
        if cls=='cls':
            printacc(pidx, plabel, Y_unlabel)
            print('same number')
            printacc2(pidx, plabel, Y_unlabel)
        else:
            printacc(pidx, kl_yhat, Y_unlabel)
            print('same number')
            printacc2(pidx, kl_yhat, Y_unlabel)
        return plabel, pidx, tpreds
    cls_plabel, cls_pidx, cls_preds = getidx(cls_preds, 'cls')
    kl_plabel, kl_pidx, kl_preds = getidx(kl_preds, 'kl')

    print('percentage of consistent prediction between two network: {:.3f}'.format(np.mean(kl_yhat==cls_plabel)))
    cst_idx = np.where(kl_yhat==cls_plabel)[0]
    print('acc of consistent part of the two networks: {:.3f}'.format(np.mean(kl_yhat[cst_idx]==Y_unlabel[cst_idx])))
    lb1 = cls_plabel[cst_idx]
    lb2 = kl_yhat[cst_idx]

    assert(np.sum(lb1!=lb2)==0)

    p1 = cls_preds[cst_idx]
    p2 = kl_preds[cst_idx]

    i1 = np.argsort(p1)
    i2 = np.argsort(p2)

    print('accs of consistent part of cls network')
    printacc(i1, lb1, Y_unlabel[cst_idx])
    print('same number')
    printacc2(i1, lb1, Y_unlabel[cst_idx])
    print('accs of consistent part of kl network')
    printacc(i2, lb2, Y_unlabel[cst_idx])
    print('same number')
    printacc2(i2, lb2, Y_unlabel[cst_idx])
    return cst_idx, lb1, lb2, i1, i2

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
        if len(sys.argv) > 2:
            kwargs['n'] = int(sys.argv[2])
        else:
            kwargs['n'] = 3
        if len(sys.argv) > 3:
            kwargs['model'] = sys.argv[3]
        # Load the dataset
        #kwargs['n'] = 3
        print("Loading data...")
        lb_num = int(sys.argv[1])
        data = load_svhn()
        data = split_data(data, lb_num, 5000)

        X_label = data['X_label']
        Y_label = data['Y_label']
        X_unlabel = data['X_unlabel']
        Y_unlabel = data['Y_unlabel']


        kwargs['cls_num'] = 10

        thres = [0.9,0.7,0.5,0.3,0,0]#
        #thres = [5000, 10000, 15000, 20000, 25000, 30000, 0, 0]
        kwargs['num_epochs'] = 150
        kwargs['strategy'] = 'momentum'
        kwargs['lb_num'] = lb_num
        model = 'model_mt_%d_%d.npz' % (lb_num,kwargs['n'])
        if os.path.exists(model):
            kwargs['model'] = model
        else:
            kwargs['model'] = None

        for i in thres:
            print('thres = ', i)

            input_var, cls_network, kl_network, val_fn, indexes = main(data, **kwargs)
            ###########################################################################
#            net=lasagne.layers.get_all_layers(cls_network)
#            cls_gp = net[-2]
#            net=lasagne.layers.get_all_layers(kl_network)
#            kl_gp = net[-2]
#
            cls_preds, kl_preds, kl_yhat, indexes, cm = get_mtresults(X_unlabel, Y_unlabel, 'unlabel', val_fn, indexes = indexes)
#            #prediction acc on unlabeled data using the trained network
#
            comp_preds(cls_preds, kl_preds, kl_yhat, Y_unlabel)
#
            unlabel_num = Y_unlabel.shape[0]
            def getidx(preds):
                plabel = np.argmax(preds, axis=1)
                tpreds = preds[np.arange(unlabel_num), plabel]
                pidx = np.argsort(tpreds)
                return plabel, pidx, tpreds
            cls_plabel, cls_pidx, cls_preds = getidx(cls_preds)
#            cst_idx, lb1, lb2, i1, i2 = comp_preds(cls_preds, kl_preds, kl_yhat, Y_unlabel)
#            print('len of cst_idx: ', cst_idx.shape[0] )

#            def getFeats(X, Y, cls, kl, input_var):
#                cls_feat = getFeat(X, Y, cls, input_var)
#                kl_feat = getFeat(X, Y, kl, input_var)
#                return cls_feat, kl_feat
##            lb_cls_feat, lb_kl_feat = getFeats(X_label, Y_label, cls_gp, kl_gp, input_var)
##            ulb_cls_feat, ulb_kl_feat = getFeats(X_unlabel, Y_unlabel, cls_gp, kl_gp, input_var)
#            lb_cls_feat, lb_kl_feat = getFeats(X_label, Y_label, cls_network, kl_network, input_var)
#            ulb_cls_feat, ulb_kl_feat = getFeats(X_unlabel, Y_unlabel, cls_network, kl_network, input_var)
            ###########average################
#            lb_kl_center = []
#            for c in np.arange(10):
#                tmp = (Y_label == c)
#                lb_kl_center.append(np.mean(lb_kl_feat[tmp], axis=0))
#            lb_kl_center = np.vstack(lb_kl_center)
            ###########average distant to other points in the same class###########
#            lb_kl_center = []
#            for c in np.arange(10):
#                tmpi = Y_label == c
#                c_feat = lb_kl_feat[tmpi]
#                c_dist = cdist(c_feat, c_feat)
#                c_dist = np.sum(c_dist, 1)
#                lb_kl_center.append(lb_kl_feat[tmpi][np.argmin(c_dist)])
#            lb_kl_center = np.vstack(lb_kl_center)
            ##############highest confidence###############
#            lb_kl_center = []
#            lb_kl_preds = getFeat(X_label, Y_label, kl_network, input_var)
#            for c in np.arange(10):
#                mc = indexes[c,1]
#                tmpi = (Y_label == c)
#                c_feat = lb_kl_preds[tmpi]
#                c_feat = c_feat[:,mc]
#                lb_kl_center.append(lb_kl_feat[tmpi][np.argmax(c_feat)])
#            lb_kl_center = np.vstack(lb_kl_center)
#            ##################################################
#            kl_dist = np.sum((lb_kl_center - ulb_kl_feat.reshape((unlabel_num, 1, -1)))**2, 2)
#            min_idx = np.argmin(kl_dist, axis=1)
##            for j,mi in enumerate(min_idx):
##                min_idx[j] = indexes[mi,1]
#
#            tidx = int(unlabel_num * i)
##            if i == 0:
##                i = cst_idx.shape[0]
#            #tidx = cst_idx.shape[0] - i
#            aidx = cls_pidx[tidx:]
#            a = cls_plabel[aidx]
#            b = min_idx[aidx]
#            bidx = (a == b)
#            print('consistent number: ', np.sum(bidx))
#            print('acc of consistent part: ', np.mean(cls_plabel[aidx][bidx] == Y_unlabel[aidx][bidx]))
#            X_train = np.vstack((X_label, X_unlabel[aidx][bidx]))
#            Y_train = np.concatenate((Y_label, cls_plabel[aidx][bidx]))
            tidx = int(unlabel_num * i)
#            if i == 0:
#                i = cst_idx.shape[0]
            #tidx = cst_idx.shape[0] - i
            aidx = cls_pidx[tidx:]
            X_train = np.vstack((X_label, X_unlabel[aidx]))
            Y_train = np.concatenate((Y_label, cls_plabel[aidx]))
            data['X_train'] = X_train
            data['Y_train'] = Y_train.astype('int32')

            kwargs['model'] = None
#            data_cls['X_train'] = X_train
#            data_cls['Y_train'] = Y_train.astype('int32')
#
#            aidx = i2[tidx:]
#
#            X_train = np.vstack((X_label, X_unlabel[cst_idx][aidx]))
#            Y_train = np.concatenate((Y_label, lb2[aidx]))
#            data_kl['X_train'] = X_train
#            data_kl['Y_train'] = Y_train.astype('int32')
