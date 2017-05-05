# -*- coding: utf-8 -*-
#3 cifar10_deep_residual_model.npz
from __future__ import print_function
import sys
import os
import time
import string
import random
import pickle
import copy
#import cv2

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.use('TkAgg')

sys.path.append('../')
# for the larger networks (n>=9), we need to adjust pythons recursion limit
sys.setrecursionlimit(10000)
from utils import get_mtresults, architecture_string,\
    printacc, printacc2, getFeat, getkl, load_mit, iou

import theano
import theano.tensor as T
from lasagne.objectives import aggregate
import lasagne

from scipy.spatial.distance import cdist
from cotrain_model import build_cnn
from losses import get_klloss

# ############################## Main program ################################


def main(data, cls_num, dim, pos_ratio, w=None, increase=True, n=2, num_epochs=500, strategy='momentum', model=None, thres=None,lb_num=None):



    x_train = data['training']
    x_val = data['val']


    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    w_var = T.vector('w')

    num_boxes_train = len(x_train['image_ids'])
    pos_idx = (x_train['boxes'][:,-1]==1).nonzero()[0]
    print(num_boxes_train, 'pos:{}'.format(pos_idx.shape[0]))
    # Create neural network model
    print("Building model and compiling functions...")
    cls_network, kl_network = build_cnn(cls_num, input_var, n, dim)

    epsilon = 1e-35
    margin = 0.5
    batchsize = target_var.shape[0]

    cls_prediction = lasagne.layers.get_output(cls_network)
   # cls_prediction = T.clip(cls_prediction, 1e-7, 1-(1e-7))
    kl_prediction = lasagne.layers.get_output(kl_network)

    # add weight decay
    all_layers = lasagne.layers.get_all_layers([cls_network, kl_network])
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0005

    cls_loss = lasagne.objectives.categorical_crossentropy(cls_prediction, target_var)
    cls_loss = aggregate(cls_loss, w_var)

    kl_prediction = kl_prediction + epsilon

    kl_loss = get_klloss(batchsize, margin, kl_prediction, target_var)
    loss = l2_penalty + 0*kl_loss + cls_loss

    lr = 0.1
    sh_lr = theano.shared(lasagne.utils.floatX(lr))
    params = lasagne.layers.get_all_params([cls_network, kl_network], trainable=True)
    if strategy=='momentum':
        updates = lasagne.updates.momentum(loss, params, learning_rate=sh_lr, momentum=0.9)#nesterov_
    else:
        updates = lasagne.updates.adam(loss, params, learning_rate=1e-4)

    train_fn = theano.function([input_var, target_var, w_var], [loss, cls_prediction, kl_prediction],
                               updates=updates, allow_input_downcast=True)#, w_var

    # Create a loss expression for validation/testing
    test_cls_prediction = lasagne.layers.get_output(cls_network, deterministic=True)
    test_cls_prediction = T.clip(test_cls_prediction, 1e-7, 1-1e-7)
    test_cls_loss = lasagne.objectives.categorical_crossentropy(test_cls_prediction, target_var)
    test_cls_loss = test_cls_loss.mean()

    test_kl_prediction = lasagne.layers.get_output(kl_network, deterministic=True)
    test_kl_prediction = test_kl_prediction + epsilon
    test_kl_loss = get_klloss(batchsize, margin, test_kl_prediction, target_var)

    test_loss = l2_penalty + test_kl_loss + test_cls_loss

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_cls_prediction, test_kl_prediction],
                             allow_input_downcast=True)

    print("number of parameters in model: %d" % lasagne.layers.count_params([cls_network, kl_network], trainable=True))
    print(architecture_string([cls_network, kl_network]))
    if model is not None and os.path.exists(model):
        # load network weights from model file
        with np.load(model) as f:
             param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        with np.load('weighted_indexes_{}'.format(model)) as f:
            indexes = f['arr_0']
        lasagne.layers.set_all_param_values([cls_network, kl_network], param_values)
    else:
        # launch the training loop
        print("Starting training...")
        for epoch in np.arange(num_epochs):
            print("Epoch {} of {}:".format(epoch + 1, num_epochs))

            cls_preds, kl_preds, my_hat, indexes, cm = get_mtresults(x_train,\
                'train','train', train_fn, cls_num, dim, pos_ratio, True, w=w, batchsize=128)

            cls_preds, kl_preds, my_hat, indexes, cm = get_mtresults(x_val,\
                'val', 'train',val_fn, cls_num, dim, pos_ratio, indexes = indexes)

            if (epoch+1) == 30 or (epoch+1) == 50 or (epoch+1)==70:
                new_lr = sh_lr.get_value() * 0.1
                print("New LR:"+str(new_lr))
                sh_lr.set_value(lasagne.utils.floatX(new_lr))
        # dump the network weights to a file :
        if model is not None:
            np.savez(model, *lasagne.layers.get_all_param_values([cls_network, kl_network]))
            np.savez('weighted_indexes_{}'.format(model), indexes)

    return input_var, cls_network, kl_network, val_fn, indexes

def comp_preds(cls_preds, kl_preds, kl_yhat, y_unlabel):
    unlabel_num = kl_preds.shape[0]

    def getidx(preds, cls):
        plabel = np.argmax(preds, axis=1)
        tpreds = preds[np.arange(unlabel_num), plabel]
        pidx = np.argsort(tpreds)
        print('prediction of ', cls)
        if cls=='cls':
            printacc(pidx, plabel, y_unlabel)
            #print('same number')
            #printacc2(pidx, plabel, y_unlabel)
        else:
            printacc(pidx, kl_yhat, y_unlabel)
            #print('same number')
            #printacc2(pidx, kl_yhat, y_unlabel)
        return plabel, pidx, tpreds
    cls_plabel, cls_pidx, cls_preds = getidx(cls_preds, 'cls')
    kl_plabel, kl_pidx, kl_preds = getidx(kl_preds, 'kl')

    print('percentage of consistent prediction between two network: {:.3f}'.format(np.mean(kl_yhat==cls_plabel)))
    cst_idx = np.where(kl_yhat==cls_plabel)[0]
    print('acc of consistent part of the two networks: {:.3f}'.format(np.mean(kl_yhat[cst_idx]==y_unlabel[cst_idx])))
    lb1 = cls_plabel[cst_idx]
    lb2 = kl_yhat[cst_idx]

    assert(np.sum(lb1!=lb2)==0)

    p1 = cls_preds[cst_idx]
    p2 = kl_preds[cst_idx]

    i1 = np.argsort(p1)
    i2 = np.argsort(p2)

#    print('accs of consistent part of cls network')
#    printacc(i1, lb1, y_unlabel[cst_idx])
#    print('same number')
#    printacc2(i1, lb1, y_unlabel[cst_idx])
#    print('accs of consistent part of kl network')
#    printacc(i2, lb2, y_unlabel[cst_idx])
#    print('same number')
#    printacc2(i2, lb2, y_unlabel[cst_idx])
    return cst_idx, lb1, lb2, i1, i2
def getidx(preds):
    unlabel_num = preds.shape[0]
    plabel = np.argmax(preds, axis=1)
    tpreds = preds[np.arange(unlabel_num), plabel]
    pidx = np.argsort(tpreds)
    return plabel, pidx, tpreds



def nms(candidate, nms_thres):
    image_ids = candidate['image_ids']
    boxes = candidate['boxes']
    uniq_image_ids = np.unique(image_ids)
    keep_idx = np.ones(image_ids.shape[0])

    for iid in uniq_image_ids:
        idx = (image_ids==iid).nonzero()[0]
        for x in np.arange(0,len(idx)):
            if keep_idx[idx[x]]==0:
                continue
            loc_x = boxes[idx[x],2:6]
            for y in np.arange(x+1,len(idx)):
                loc_y = boxes[idx[y],2:6]
                if iou(loc_x, loc_y)>=nms_thres:
                    keep_idx[idx[y]] = 0
    keep_idx = (keep_idx==1).nonzero()[0]
    candidate['image_ids'] = image_ids[keep_idx]
    candidate['boxes'] = boxes[keep_idx]
    return candidate

def calculate_det(data, dataset, fn, cls_num, dim, pos_ratio, nms_thres,iou_thres, indexes):
    cls_preds, kl_preds, kl_yhat, indexes, cm = get_mtresults(data, dataset, \
                    'test', val_fn, cls_num, dim, pos_ratio, indexes = indexes)

#    cls_plabel = np.argmax(cls_preds, axis=1)
    cls_preds = kl_preds
    pos_idx = 1
    for pred, gt in indexes:
        if pred == 1:
            pos_idx = gt
            break
    pos_preds = cls_preds[:,pos_idx]
    cls_pidx = np.argsort(pos_preds)
    cls_pidx = cls_pidx[::-1]

    boxes = data['boxes'][cls_pidx]
    boxes = np.column_stack((boxes,pos_preds[cls_pidx]))
    image_ids = np.array(data['image_ids'])
    image_ids = image_ids[cls_pidx]

#    boxes = boxes[np.arange(boxes.shape[0]-1,-1,-1)]
#    image_ids = image_ids[np.arange(boxes.shape[0]-1,-1,-1)]



    gt = {}
    gt_idx = (boxes[:,0]==1).nonzero()[0]
    gt['image_ids'] = image_ids[gt_idx]

    gt['boxes'] = boxes[gt_idx]
    gt['match'] = np.zeros(gt['boxes'].shape[0], dtype=np.int8)
    print('total gts number:{}'.format(gt['boxes'].shape[0]))

    candidate = {}
    candidate_idx = (boxes[:,0]==0).nonzero()[0]
    candidate['image_ids'] = image_ids[candidate_idx]

    candidate['boxes'] = boxes[candidate_idx]

    print('total candidates number:{}'.format(candidate['boxes'].shape[0]))
    candidate = nms(candidate, nms_thres)
    print('nms candidates number:{}'.format(candidate['boxes'].shape[0]))


    num_images = (np.unique(candidate['image_ids'])).shape[0]
    num_detects = candidate['image_ids'].shape[0]
    total_num_gts = gt['image_ids'].shape[0]

    fp = np.zeros(num_detects,dtype=np.float)
    tp = np.zeros(num_detects,dtype=np.float)


    for did in np.arange(0,num_detects):
        dbox = candidate['boxes'][did,2:6]
        iid = candidate['image_ids'][did]
        gidx = (gt['image_ids']==iid).nonzero()[0]
        max_iou = 0
        max_id = -1
        for gid in gidx:
            gbox = gt['boxes'][gid,2:6]
            tiou = iou(dbox, gbox)
            if tiou>max_iou:
                max_iou = tiou
                max_id = gid
#        print(max_iou)
        if max_iou>=iou_thres and gt['match'][max_id]==0:
            tp[did] = 1
            gt['match'][max_id] = 1
        else:
            fp[did] = 1

    fppi = np.cumsum(fp)/num_images
    drate = np.cumsum(tp)/total_num_gts

    idx = (fppi<=10).nonzero()[0]

    plt.plot(fppi[idx], drate[idx])
    plt.show()

    idx = (fppi<=1).nonzero()[0]
    print('when fppi=1, recall={}'.format(drate[idx[-1]]))
    ret = {}
    ret['images'] = data['images']
    ret['gt'] = gt
    ret['candidate'] = candidate

    return ret, drate, fppi
def visualize(detections, drate, fppi):

    can_boxes = detections['candidate']['boxes']
    can_image_ids = detections['candidate']['image_ids']
    gt_boxes = detections['gt']['boxes']
    gt_image_ids = detections['gt']['image_ids']


    uniq_image_ids = np.unique(can_image_ids)
    idx = (fppi<=1).nonzero()[0]
    can_boxes = can_boxes[idx]
    can_image_ids = can_image_ids[idx]
    for iid in uniq_image_ids:
        cid = (can_image_ids==iid).nonzero()[0]
        gid = (gt_image_ids==iid).nonzero()[0]


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

        if len(sys.argv) > 3:
            kwargs['n'] = int(sys.argv[3])
        else:
            kwargs['n'] = 1
        if len(sys.argv) > 4:
            kwargs['model'] = sys.argv[4]
        #kwargs['n'] = 3
        # Load the dataset
        print("Loading data...")


        pos_neg_thres = [0.5, 0.5]
        data = load_mit(pos_neg_thres)
        #data = split_data(data, lb_num, 5000)


        x_train = data['train']
        x_val = data['val']
        x_unlabel = data['unlabel']
        x_test = data['test']

#        training_data = {}
#        training_data['images'] = x_train['images']
#        training_data['image_ids'] = x_train['image_ids']
#        training_data['boxes'] = x_train['boxes']
        training_data = copy.deepcopy(x_train)
        data['training'] = training_data

        nms_thres = 0.3
        iou_thres = 0.5
        kwargs['cls_num'] = 2
#        kwargs['dim'] = (3,224,224)
        kwargs['dim'] = (3,128,64)

        kwargs['pos_ratio'] = 0.5
        dim = kwargs['dim']
        pos_ratio = kwargs['pos_ratio']

        num_train = len(x_train['image_ids'])
        w = np.ones(num_train,dtype=np.float32)

        kwargs['num_epochs'] = 80
        kwargs['strategy'] = 'momentum'
        kwargs['w'] = w
        lb_num=2000
        kwargs['lb_num'] = lb_num



        thres = [1,0.9,0.7,0.5,0.3,0]
        slct_num = [-1, 200, 0]

        for spl_iter in np.arange(len(thres)):
            i = thres[spl_iter]
            print('thres = ', i)
            kwargs['model'] = 'model_nocls_epochs80_spl_noweight_v3_%.2f.npz' %(i)
            kwargs['thres'] = i
            input_var, cls_network, kl_network, val_fn, indexes = main(data, **kwargs)
            net=lasagne.layers.get_all_layers(cls_network)
            cls_gp = net[-2]
            net=lasagne.layers.get_all_layers(kl_network)
            kl_gp = net[-2]

#            cls_preds, kl_preds, kl_yhat, indexes, cm = get_mtresults(training_data, 'train', \
#                    'test', val_fn, kwargs['cls_num'], dim, pos_ratio, indexes = indexes)
#

            cls_preds, kl_preds, kl_yhat, indexes, cm = get_mtresults(x_unlabel,'unlabel',
                    'test', val_fn, kwargs['cls_num'], dim, pos_ratio, indexes = indexes)

            #prediction acc on unlabeled data using the trained network
            #kwargs['model'] = None
            boxes_unlabel = x_unlabel['boxes']
            img_ids_unlabel = x_unlabel['image_ids']
            img_ids_unlabel = np.array(img_ids_unlabel)
            y_unlabel = boxes_unlabel[:,-1]
            gt_unlabel = boxes_unlabel[:,0]

            cls_plabel, cls_pidx, cls_preds2 = getidx(cls_preds)
            pos_index = (cls_plabel==1).nonzero()[0]
            comp_preds(cls_preds[pos_index], kl_preds[pos_index],\
                       kl_yhat[pos_index], y_unlabel[pos_index])
            comp_preds(cls_preds, kl_preds, kl_yhat, y_unlabel)

            assert(cls_preds.shape[0]==y_unlabel.shape[0])

            if True:
#                train_detects,train_drate,train_fppi = calculate_det(training_data,'train',\
#                        val_fn, kwargs['cls_num'],dim, pos_ratio,nms_thres,iou_thres, indexes)
                test_detects,test_drate,test_fppi = calculate_det(x_test,'test',\
                        val_fn,kwargs['cls_num'],dim, pos_ratio,nms_thres,iou_thres, indexes)
#                visualize(test_detects, test_drate, test_fppi)



            def getFeats(X, dim, pos_ratio, cls, kl, input_var):
                cls_feat = getFeat(X, dim, pos_ratio, cls, input_var)
                kl_feat = getFeat(X, dim, pos_ratio, kl, input_var)
                return cls_feat, kl_feat


            sorted_cls_plabel = cls_plabel[cls_pidx]
            kl_preds = kl_preds[cls_pidx]
            gt_unlabel = gt_unlabel[cls_pidx]
            boxes_unlabel = boxes_unlabel[cls_pidx]
            img_ids_unlabel = img_ids_unlabel[cls_pidx]
            non_gt_idx = (gt_unlabel==0).nonzero()[0]

            sorted_cls_plabel = sorted_cls_plabel[non_gt_idx]
            kl_preds = kl_preds[non_gt_idx]
            gt_unlabel = gt_unlabel[non_gt_idx]
            boxes_unlabel = boxes_unlabel[non_gt_idx]
            img_ids_unlabel = img_ids_unlabel[non_gt_idx]

            pos_idx = (sorted_cls_plabel==1).nonzero()[0]
            pos_unlabel_num = pos_idx.shape[0]

            if spl_iter+1==len(thres):
                break

            slct_startidx = int(pos_unlabel_num * thres[spl_iter+1])
#            slct_startidx = max(slct_startidx, pos_unlabel_num-slct_num[spl_iter+1])


            pos_idx_slct = pos_idx[slct_startidx:]
            kl_preds = kl_preds[pos_idx_slct]

            if pos_idx_slct.shape[0]==0:
                print('null selected pseudo samples')
                kwargs['model'] = None
            else:
                print('{} selected pseudo samples'.format(pos_idx_slct.shape[0]))
                lb_cls_feat, lb_kl_feat = getFeats(x_train, dim, pos_ratio, cls_network, kl_network, input_var)

                y_label = x_train['boxes'][:,-1]
                lb_pos_idx = (y_label==1).nonzero()[0]
                lb_kl_feat = lb_kl_feat[lb_pos_idx]

                print('using kl dist')
                pw = np.zeros(pos_idx_slct.shape[0])

                kl_dist = getkl(kl_preds, lb_kl_feat)
                beta = 1
                if len(sys.argv)>5:
                    beta = float(sys.argv[5])
                kl_dist = np.exp(-beta*kl_dist)

                pw = np.mean(kl_dist, 1)
    ###########################################################

                gamma = 0.1
                if len(sys.argv)>2:
                    gamma = float(sys.argv[2])
                print(gamma)
                pw = pw / np.sum(pw) * pos_idx_slct.shape[0] * gamma
                pw = np.ones(pos_idx_slct.shape)

                kwargs['w'] = np.concatenate((w, pw))

#                training_data = x_train
                training_data = copy.deepcopy(x_train)
#                training_data['images'] = x_train['images']
#                training_data['image_ids'] = x_train['image_ids']
#                training_data['boxes'] = x_train['boxes']
                for idx in pos_idx_slct:
                    img_id = img_ids_unlabel[idx]
                    training_data['image_ids'].append(img_id)
                    training_data['images'][img_id] = x_unlabel['images'][img_id]
                slct_unlabel_boxes = boxes_unlabel[pos_idx_slct]
                slct_unlabel_boxes[:,-1] = 1
                training_data['boxes'] = np.concatenate((x_train['boxes'],slct_unlabel_boxes),axis=0)

                data['training'] = training_data
                print('length of x_train: {}'.format(x_train['boxes'].shape[0]))
                print('length of w: {}'.format(kwargs['w'].shape[0]))
                print('length of training data: {}'.format(data['training']['boxes'].shape[0]))
                assert kwargs['w'].shape[0]==data['training']['boxes'].shape[0]
#                assert kwargs['w'].shape[0]==len(data['training']['image_ids'])





