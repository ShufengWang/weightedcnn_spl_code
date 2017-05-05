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
from utils import get_mtresults, architecture_string, printacc, printacc2, load_mit, iou

import theano
import theano.tensor as T
from lasagne.objectives import aggregate
import lasagne

from scipy.spatial.distance import cdist
from cotrain_model import build_cnn
from cotrain_model import build_alexnet

# ############################## Main program ################################


def main(data, cls_num, dim, pos_ratio, w=None, increase=True, n=2, num_epochs=500, strategy='momentum', last_model=None, model=None, thres=None,lb_num=None):


    x_train = data['training']
    x_val = data['val']


    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    w_var = T.vector('w')

    neg_idx = (x_train['boxes'][:,-1]==0).nonzero()[0]
    pos_idx = (x_train['boxes'][:,-1]==1).nonzero()[0]
    print('neg:{}'.format(neg_idx.shape[0]), 'pos:{}'.format(pos_idx.shape[0]))
    # Create neural network model
    print("Building model and compiling functions...")
    network = build_alexnet(cls_num, input_var, dim)
    fc8 = network['fc8']
    print(architecture_string(fc8))

    batchsize = target_var.shape[0]

    prediction = lasagne.layers.get_output(fc8)

    # add weight decay
    all_layers = lasagne.layers.get_all_layers(fc8)
    l2_penalty = lasagne.regularization.regularize_layer_params(
                    all_layers, lasagne.regularization.l2) * 0.0005

    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    #loss = loss.mean()
    loss = aggregate(loss, w_var)
    loss = loss + l2_penalty

    acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var), dtype=theano.config.floatX)

    lr = 0.001
    sh_lr = theano.shared(lasagne.utils.floatX(lr))
    params = lasagne.layers.get_all_params(fc8, trainable=True)
    if strategy=='momentum':
        updates = lasagne.updates.momentum(loss, params, learning_rate=sh_lr, momentum=0.9)#nesterov_
    else:
        updates = lasagne.updates.adam(loss, params, learning_rate=1e-4)

    train_fn = theano.function([input_var, target_var, w_var], [loss, acc, prediction],
                               updates=updates, allow_input_downcast=True)

    # Create a loss expression for validation/testing
    test_prediction = lasagne.layers.get_output(fc8, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                    dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc,
        test_prediction], allow_input_downcast=True)

    print("number of parameters in model: %d" % lasagne.layers.count_params(fc8, trainable=True))
    # print(architecture_string(fc8))

    if model is not None and os.path.exists(model):
        # load network weights from model file
        with np.load(model) as f:
             param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        #with np.load('weighted_indexes_{}'.format(model)) as f:
        #    indexes = f['arr_0']
        lasagne.layers.set_all_param_values(fc8, param_values)
    else:
        if last_model is None:
            # first loop of training, copy model from caffe

            caffe_root = '/home/dl/caffe/'
            sys.path.insert(0, caffe_root + 'python')
            import caffe

            caffe.set_mode_gpu()
            caffe_net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)

            layers_caffe = dict(zip(list(caffe_net._layer_names), caffe_net.layers))
            for name, layer in network.items():
                try:
                    if name == 'conv2':
                        W = layers_caffe[name].blobs[0].data[:,:,::-1,::-1]
                        b = layers_caffe[name].blobs[1].data

                        network['conv2_part1'].W.set_value(W[0:128,:,:,:])
                        network['conv2_part1'].b.set_value(b[0:128])
                        network['conv2_part2'].W.set_value(W[128:,:,:,:])
                        network['conv2_part2'].b.set_value(b[128:])
                    elif name == 'conv4':
                        W = layers_caffe[name].blobs[0].data[:,:,::-1,::-1]
                        b = layers_caffe[name].blobs[1].data

                        network['conv4_part1'].W.set_value(W[0:192,:,:,:])
                        network['conv4_part1'].b.set_value(b[0:192])
                        network['conv4_part2'].W.set_value(W[192:,:,:,:])
                        network['conv4_part2'].b.set_value(b[192:])
                    elif name == 'conv5':
                        W = layers_caffe[name].blobs[0].data[:,:,::-1,::-1]
                        b = layers_caffe[name].blobs[1].data

                        network['conv5_part1'].W.set_value(W[0:128,:,:,:])
                        network['conv5_part1'].b.set_value(b[0:128])
                        network['conv5_part2'].W.set_value(W[128:,:,:,:])
                        network['conv5_part2'].b.set_value(b[128:])
                    elif name == 'fc6' or name == 'fc7':
                        layer.W.set_value(np.transpose(layers_caffe[name].blobs[0].data))
                        layer.b.set_value(layers_caffe[name].blobs[1].data)
                    elif name != 'fc8':
                        layer.W.set_value(layers_caffe[name].blobs[0].data[:,:,::-1,::-1])
                        layer.b.set_value(layers_caffe[name].blobs[1].data)
                except AttributeError:
                    continue
                except KeyError:
                    continue
            print(architecture_string(fc8))
        elif os.path.exists(last_model):
            with np.load(last_model) as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            #with np.load('weighted_indexes_{}'.format(last_model)) as f:
            #    indexes = f['arr_0']
            lasagne.layers.set_all_param_values(fc8, param_values)

        print("Starting training...")
        for epoch in np.arange(num_epochs):
            print("Epoch {} of {}:".format(epoch + 1, num_epochs))

            preds = get_mtresults(x_train,\
                'train','train', train_fn, cls_num, dim, pos_ratio, True, w=w, batchsize=128)

            preds = get_mtresults(x_val,\
                'val', 'train', val_fn, cls_num, dim, pos_ratio)

            if (epoch+1) == 10 or (epoch+1) == 20 or (epoch+1)==30:
                new_lr = sh_lr.get_value() * 0.1
                print("New LR:"+str(new_lr))
                sh_lr.set_value(lasagne.utils.floatX(new_lr))
        # dump the network weights to a file :
        if model is not None:
            np.savez(model, *lasagne.layers.get_all_param_values(fc8))
            #np.savez('weighted_indexes_{}'.format(model), indexes)

    return input_var, fc8, val_fn

def comp_preds(preds, y_unlabel):

    plabel, pidx, tpreds = get_label_idx(preds)
    print('prediction')
    printacc(pidx, plabel, y_unlabel)
    #print('same number')
    #printacc2(pidx, plabel, y_unlabel)

def get_label_idx(preds):
    num = preds.shape[0]
    plabel = np.argmax(preds, axis=1)
    tpreds = preds[np.arange(num), plabel]
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

def calculate_det(data, dataset, fn, cls_num, dim, pos_ratio, nms_thres,iou_thres):
    preds = get_mtresults(data, dataset, \
                    'test', val_fn, cls_num, dim, pos_ratio)

    pos_preds = preds[:,1]
    sorted_pidx = np.argsort(pos_preds)
    sorted_pidx = sorted_pidx[::-1]

    boxes = data['boxes'][sorted_pidx]
    boxes = np.column_stack((boxes,pos_preds[sorted_pidx]))
    image_ids = np.array(data['image_ids'])
    image_ids = image_ids[sorted_pidx]


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

    #plt.plot(fppi[idx], drate[idx])
    #plt.show()

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

def shake(boxes, img_ids, pw):
    shaked_boxes = []
    shaked_img_ids = []
    shaked_w = []
    for dx in np.arange(-2,3,2):
        for dy in np.arange(-2,3,2):
            t_boxes = boxes
            t_boxes[:,2] = t_boxes[:,2]+dx
            t_boxes[:,3] = t_boxes[:,3]+dy
            t_boxes[:,4] = t_boxes[:,4]+dx
            t_boxes[:,5] = t_boxes[:,5]+dy
            shaked_boxes.append(t_boxes)
            shaked_img_ids.append(img_ids)
            shaked_w.append(pw)

    shaked_boxes = np.vstack(shaked_boxes)
    shaked_img_ids = np.hstack(shaked_img_ids)
    shaked_w = np.hstack(shaked_w)
    return shaked_boxes, shaked_img_ids,shaked_w


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains AlexNet on pedestrian-detection dataset using Lasagne.")
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

        pos_neg_thres = [0.3, 0.7]
        data = load_mit(pos_neg_thres)

        x_train = data['train']
        x_val = data['val']
        x_unlabel = data['unlabel']
        x_test = data['test']

        training_data = copy.deepcopy(x_train)
        data['training'] = training_data

        w_lambda = 0.5
        confident_thres = 0.9
        nms_thres = 0.3
        iou_thres = 0.5
        cls_num = 2
        kwargs['cls_num'] = cls_num
#        kwargs['dim'] = (3,224,224)
        kwargs['dim'] = (3,227,227)

        kwargs['pos_ratio'] = 0.5
        dim = kwargs['dim']
        pos_ratio = kwargs['pos_ratio']

        num_train = len(x_train['image_ids'])
        w = np.ones(num_train,dtype=np.float32)

        kwargs['num_epochs'] = 30
        kwargs['strategy'] = 'momentum'
        kwargs['w'] = w
        lb_num=2000
        kwargs['lb_num'] = lb_num

        thres = [1,0.9,0.8,0.7, 0.6, 0.5]
        slct_num = [0,100,200, 300, 400, 500]
        drates = np.zeros(len(thres), dtype=np.float)

        for spl_iter in np.arange(len(thres)):
            i = thres[spl_iter]
            print('thres = ', i)
            if spl_iter == 0:
                kwargs['last_model'] = None
            else:
                kwargs['last_model'] = None #'model_weightedcnn_spl_no_pretrain_%.2f.npz' %(thres[spl_iter-1])
            kwargs['model'] = 'wcnn_spl_less_train_shake_v1_r%d.npz' %(slct_num[spl_iter])
            kwargs['thres'] = i
            input_var, network, val_fn = main(data, **kwargs)
            net=lasagne.layers.get_all_layers(network)

            tra_preds = get_mtresults(x_train, 'train',
                    'test', val_fn, kwargs['cls_num'], dim, pos_ratio)
            preds = get_mtresults(x_unlabel, 'unlabel',
                    'test', val_fn, kwargs['cls_num'], dim, pos_ratio)
            if True:
#               train_detects,train_drate,train_fppi = calculate_det(training_data,'train',\
#                        val_fn, kwargs['cls_num'],dim, pos_ratio,nms_thres,iou_thres)
                test_detects,test_drate,test_fppi = calculate_det(x_test,'test',\
                        val_fn,kwargs['cls_num'],dim, pos_ratio,nms_thres,iou_thres)
                idx = (test_fppi<=1).nonzero()[0]
                drates[spl_iter] = test_drate[idx[-1]]
#                visualize(test_detects, test_drate, test_fppi)



            tra_scores = tra_preds[:,1]
            unlabel_scores = preds[:,1]
            unlabel_w = np.array(unlabel_scores)

            eps = 1e-10
            max_w = 0
            for ii in np.arange(unlabel_scores.shape[0]):
                s = unlabel_scores[ii]
                unlabel_w[ii] = max(1/(s-tra_scores+eps))
                max_w = max(max_w, unlabel_w[ii])
            unlabel_w = unlabel_w/max_w

            #prediction acc on unlabeled data using the trained network
            #kwargs['model'] = None
            boxes_unlabel = x_unlabel['boxes']
            img_ids_unlabel = x_unlabel['image_ids']
            img_ids_unlabel = np.array(img_ids_unlabel)
            y_unlabel = boxes_unlabel[:,-1]
            gt_unlabel = boxes_unlabel[:,0]

            plabel, sorted_pidx, preds2 = get_label_idx(preds)
            pos_index = (plabel==1).nonzero()[0]
            comp_preds(preds[pos_index], y_unlabel[pos_index])

            assert(preds.shape[0]==y_unlabel.shape[0])

            sorted_plabel = plabel[sorted_pidx]
            gt_unlabel = gt_unlabel[sorted_pidx]
            boxes_unlabel = boxes_unlabel[sorted_pidx]
            img_ids_unlabel = img_ids_unlabel[sorted_pidx]
            preds2 = preds2[sorted_pidx]
            unlabel_w = unlabel_w[sorted_pidx]

            pos_idx = (sorted_plabel==1).nonzero()[0]
            sorted_plabel = sorted_plabel[pos_idx]
            gt_unlabel = gt_unlabel[pos_idx]
            boxes_unlabel = boxes_unlabel[pos_idx]
            img_ids_unlabel = img_ids_unlabel[pos_idx]
            preds2 = preds2[pos_idx]
            unlabel_w = unlabel_w[pos_idx]

            confident_idx = (preds2>=confident_thres).nonzero()[0]
            sorted_plabel = sorted_plabel[confident_idx]
            gt_unlabel = gt_unlabel[confident_idx]
            boxes_unlabel = boxes_unlabel[confident_idx]
            img_ids_unlabel = img_ids_unlabel[confident_idx]
            unlabel_w = unlabel_w[confident_idx]

            non_gt_idx = (gt_unlabel==0).nonzero()[0]
            pos_unlabel_num = non_gt_idx.shape[0]

            if spl_iter+1==len(thres):
                break
            print('pos_unlabel_num:{}'.format(pos_unlabel_num))
            slct_startidx = int(pos_unlabel_num * thres[spl_iter+1])
            slct_startidx = max(slct_startidx, pos_unlabel_num-slct_num[spl_iter+1])


            pos_idx_slct = non_gt_idx[slct_startidx:]

            if pos_idx_slct.shape[0]==0:
                print('null selected pseudo samples')
                kwargs['model'] = None
            else:
                print('{} selected pseudo samples'.format(pos_idx_slct.shape[0]))

                #y_label = x_train['boxes'][:,-1]
                #lb_pos_idx = (y_label==1).nonzero()[0]

                #pw = np.ones(pos_idx_slct.shape[0])
                pw = w_lambda*unlabel_w[pos_idx_slct]
                
                unlabel_w = unlabel_w[pos_idx_slct]
                img_ids_unlabel = img_ids_unlabel[pos_idx_slct]
                slct_unlabel_boxes = boxes_unlabel[pos_idx_slct]
                # last column means the pseudo label
                slct_unlabel_boxes[:,-1] = 1
                
                shaked_boxes, shaked_img_ids, shaked_pw = \
                    shake(slct_unlabel_boxes,img_ids_unlabel,unlabel_w)
                
                kwargs['w'] = np.concatenate((w, shaked_pw))

                training_data = copy.deepcopy(x_train)
                for idx in np.arange(0,shaked_pw.shape[0]):
                    img_id = shaked_img_ids[idx]
                    training_data['image_ids'].append(img_id)
                    training_data['images'][img_id] = x_unlabel['images'][img_id]
                training_data['boxes'] = np.concatenate((x_train['boxes'],shaked_boxes),axis=0)

                data['training'] = training_data
                print('length of x_train: {}'.format(x_train['boxes'].shape[0]))
                print('length of w: {}'.format(kwargs['w'].shape[0]))
                print('length of training data: {}'.format(data['training']['boxes'].shape[0]))
                assert kwargs['w'].shape[0]==data['training']['boxes'].shape[0]
#                assert kwargs['w'].shape[0]==len(data['training']['image_ids'])
        print(drates)




