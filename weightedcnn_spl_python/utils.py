# -*- coding: utf-8 -*-
import numpy as np
import random
import string
import math

from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy import io
from scipy import misc
import matplotlib.image as im

def architecture_string(layer):
    import lasagne
    model_arch = ''

    for i, layer in enumerate(lasagne.layers.get_all_layers(layer)):
        name = string.ljust(layer.__class__.__name__, 28)
        model_arch += "  %2i  %s %s  " % (i, name,
                                          lasagne.layers.get_output_shape(layer))

        if hasattr(layer, 'filter_size'):
            model_arch += str(layer.filter_size[0])
            model_arch += ' //'
        elif hasattr(layer, 'pool_size'):
            if isinstance(layer.pool_size, int):
                model_arch += str(layer.pool_size)
            else:
                model_arch += str(layer.pool_size[0])
            model_arch += ' //'
        if hasattr(layer, 'p'):
            model_arch += ' [%.2f]' % layer.p

        if hasattr(layer, 'stride'):
            model_arch += str(layer.stride[0])
        if hasattr(layer, 'learning_rate_scale'):
            if layer.learning_rate_scale != 1.0:
                model_arch += ' [lr_scale=%.2f]' % layer.learning_rate_scale
        if hasattr(layer, 'params'):
            for param in layer.params:
                if 'trainable' not in layer.params[param]:
                    model_arch += ' [NT] '

        model_arch += '\n'

    return model_arch


def load_mnist(K):
    from mnist import MNIST
    mndata = MNIST('/media/dl/data1/datasets/mnist/')
    X_test, Y_test = mndata.load_testing()
    X_train, Y_train = mndata.load_training()

    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)
    X_train = X_train.reshape((-1, 28, 28, 1)).transpose(0, 3, 1, 2)
    X_test = X_test.reshape((-1, 28, 28, 1)).transpose(0, 3, 1, 2)
    Y_train = np.asarray(Y_train)
    Y_test = np.asarray(Y_test)

    idx = np.where(Y_train<K)[0]
    X_train = X_train[idx]
    Y_train = Y_train[idx]
    idx = np.where(Y_test<K)[0]
    X_test = X_test[idx]
    Y_test = Y_test[idx]

    return dict(
        X_train=X_train.astype('float32'),
        Y_train=Y_train.astype('int32'),
        X_test = X_test.astype('float32'),
        Y_test = Y_test.astype('int32'),)

    
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_subset(root_folder, img_dir, subset, mean_value, thres, gt_flag=True):
    #fd = fopen('/media/dl/sdc1/mit_csvt/window_files/'+subset+'.txt','r')

    file_name = root_folder + 'window_files/'+subset+'.mat'

    tmp = io.loadmat(file_name)
    data = {}
    data['images'] = {}
    data['image_ids'] = []
    data['boxes'] = []

    tmp = tmp['res']
    num_iter = tmp.size
#    if subset=='unlabel':
#        num_iter = min(num_iter, 50)
#    if subset=='test':
#        num_iter = min(num_iter, 30)
    for i in np.arange(num_iter):
        x = tmp[0,i][0][0]

        img = im.imread(img_dir+x[0][0]+'.png')
#        print(img[0,0,0])
        data['images'][x[0][0]] = img - mean_value
        boxes = x[2]


        gt = boxes[:,0]
        ov = boxes[:,1]
        loc = boxes[:,2:]
        lbl = np.zeros(ov.shape,dtype=np.int32)-1

        if subset=='train': #or subset=='val':
            idx = (ov>=thres[1]).nonzero()[0]
            lbl[idx] = 1
            idx = (ov<thres[0]).nonzero()[0]
            lbl[idx] = 0
        else:
            idx = (ov>=0.5).nonzero()[0]
            lbl[idx] = 1
            idx = (ov<0.5).nonzero()[0]
            lbl[idx] = 0


#        loc = loc.astype('int32')
        boxes = np.column_stack((gt,ov,loc,lbl))

        idx = (loc>0).all(axis=1).nonzero()[0]
        boxes = boxes[idx]
        idx = (boxes[:,(2,4)]<img.shape[1]).all(axis=1).nonzero()[0]
        boxes = boxes[idx]
        idx = (boxes[:,(3,5)]<img.shape[0]).all(axis=1).nonzero()[0]
        boxes = boxes[idx]
        idx = (boxes[:,2]<boxes[:,4]).nonzero()[0]
        boxes = boxes[idx]
        idx = (boxes[:,3]<boxes[:,5]).nonzero()[0]
        boxes = boxes[idx]


        data['boxes'].append(boxes)
        for i in np.arange(boxes.shape[0]):
            data['image_ids'].append(x[0][0])

    data['boxes'] = np.vstack(data['boxes'])
    return data

def get_image_mean(root_folder, img_dir, subset):

    file_name = root_folder + 'window_files/'+subset+'.mat'

    tmp = io.loadmat(file_name)
    tmp = tmp['res']
    mean_value = 0

    for i in np.arange(tmp.size):
        x = tmp[0,i][0][0]

        img = im.imread(img_dir+x[0][0]+'.png')
        mean_value = mean_value + np.average(img)/tmp.size
    return mean_value

def iou(box1, box2):
    sx = max(box1[0], box2[0])
    ex = min(box1[2], box2[2])
    sy = max(box1[1], box2[1])
    ey = min(box1[3], box2[3])

    inter = (ex-sx)*(ey-sy)
    if sx>ex or sy > ey:
        inter = 0
    un = (box1[2]-box1[0])*(box1[3]-box1[1])+\
        (box2[2]-box2[0])*(box2[3]-box2[1])-inter
    return inter/un

def load_mit(thres):
    root_folder = '/raid/mit_spl_neurocomputing/'
    img_dir = root_folder + 'images/'

    mean_value = get_image_mean(root_folder, img_dir, 'train')
    print('images mean value is {}'.format(mean_value))
    subsets = ['train','val','unlabel','test']
    data = {}
    recall = 0
    total_gt = 0
    for subs in subsets:
        data[subs] = load_subset(root_folder, img_dir, subs, mean_value, thres)
#        boxes = data[subs]['boxes']
#        gtidx = (boxes[:,0]==1).nonzero()[0]
#        candidx = (boxes[:,0]==0).nonzero()[0]
#        flag = np.zeros(boxes.shape[0])
#        for ci in candidx:
#            cbox = boxes[ci,2:6]
#            for gi in gtidx:
#                gbox = boxes[gi,2:6]
#                if iou(gbox, cbox)>=0.5 and flag[gi]!=1:
#                    flag[gi] = 1
#                    break
#        recall = recall + np.sum(flag)
#        total_gt = total_gt + gtidx.shape[0]
#    print('total: {}, recall: {}, recall rate: {}'.format(recall, total_gt, recall/total_gt))

    return data



def iterate_minibatches(data, train_or_test, dim, batchsize, pos_ratio=0.5, shuffle=False, augment=False, epoch=1):
    image_ids = data['image_ids']
    boxes = data['boxes']
    assert len(image_ids) == boxes.shape[0]

    #num_boxes = len(image_ids['pos']) + len(image_ids['neg'])

    if train_or_test=='train':
        pos_batchsize = int(math.floor(batchsize*pos_ratio))
        neg_batchsize = batchsize - pos_batchsize
        pos_indices = (boxes[:,-1]==1).nonzero()[0]
        neg_indices = (boxes[:,-1]==0).nonzero()[0]

#        print('samples number: {} pos, {} neg'.format(pos_indices.shape[0],neg_indices.shape[0]))
#        neg_indices = neg_indices[:pos_indices.shape[0]]
        if shuffle:
            np.random.shuffle(pos_indices)
            np.random.shuffle(neg_indices)
    else:
        indices = np.arange(boxes.shape[0])
        if shuffle:
            np.random.shuffle(indices)


    for e in np.arange(epoch):
        if train_or_test=='train':
            num_iter = int(min((pos_indices.shape[0]+pos_batchsize-1)/pos_batchsize,\
                               (neg_indices.shape[0]+neg_batchsize-1)/neg_batchsize))
        else:
            num_iter = int((indices.shape[0]+batchsize-1)/batchsize)
        start_idx = [0,0]
        for iter_id in range(0, num_iter):
            if train_or_test=='train':
#                start_idx = [iter_id*pos_batchsize,iter_id*neg_batchsize]

                if start_idx[0]>=pos_indices.shape[0]:
                    start_idx[0] = 0
                    #if shuffle:
                    #    np.random.shuffle(pos_indices)
                if start_idx[1]>=neg_indices.shape[0]:
                    start_idx[1] = 0
                end_idx = [min(start_idx[0]+pos_batchsize,pos_indices.shape[0]),\
                           min(start_idx[1]+neg_batchsize,neg_indices.shape[0])]

                excerpt = np.zeros(end_idx[0]-start_idx[0]+end_idx[1]-start_idx[1], dtype=np.int32)
                excerpt[:end_idx[0]-start_idx[0]] = pos_indices[start_idx[0]:end_idx[0]]
                excerpt[end_idx[0]-start_idx[0]:] = neg_indices[start_idx[1]:end_idx[1]]
                start_idx = [start_idx[0]+pos_batchsize, start_idx[1]+neg_batchsize]

            else:
                end_idx = min(start_idx[0]+batchsize, indices.shape[0])
                excerpt = indices[start_idx[0]:end_idx]
                start_idx[0] = start_idx[0] + batchsize

            targets = boxes[excerpt,-1]

            inp_exc = np.zeros((excerpt.shape[0],dim[0],dim[1],dim[2]), dtype=np.float32)
            r = 0
            for ids in excerpt:

                img_name = image_ids[ids]
#                print(img_name)
                box = boxes[ids]
                img = data['images'][img_name]
#                print(box)
#                if dataset=='unlabel':
#                    print(box.shape, box)
                box_img = misc.imresize(img[int(box[3])+1:int(box[5])-1,int(box[2])+1:int(box[4])-1,:],[dim[1],dim[2],dim[0]])
                box_img = box_img.reshape((-1,dim[1],dim[2],dim[0])).transpose(0,3,1,2)
                inp_exc[r,:,:,:] = box_img
                r = r + 1
#            if dataset=='unlabel':
#                print(inp_exc.shape, targets.shape, excerpt.shape)
            yield inp_exc, targets, excerpt

def iterate_minibatches_probability(inputs, targets, batchsize, w):
    assert len(inputs) == len(targets)
    assert len(targets) == len(w)

    p = w/np.sum(w)
    indices = np.random.choice(np.arange(len(targets)), batchsize, False, p )

    hw = inputs.shape[3]
        # as in paper :
        # pad feature arrays with 4 pixels on each side
        # and do random cropping of 32x32
    padded = np.pad(inputs[indices],((0,0),(0,0),(4,4),(4,4)),mode='constant')
    random_cropped = np.zeros(inputs[indices].shape, dtype=np.float32)
    crops = np.random.random_integers(0,high=8,size=(batchsize,2))
    for r in range(len(indices)):
        random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+hw),crops[r,1]:(crops[r,1]+hw)]
        # random flip
        s = random.uniform(0, 1)
        if s > 0.5:
            random_cropped[r,:,:,:] = random_cropped[r, :, :, ::-1]
#                # random brightness
#                s = random.uniform(-25./255., 25./255.)
#                random_cropped[r,:,:,:] += s
#                #random  contrast
#                s = random.uniform(0.2, 1.0)
#                tmpmean = np.mean(random_cropped[r,:,:,:])
#                random_cropped[r,:,:,:] = (random_cropped[r,:,:,:] - tmpmean) * s + tmpmean
        #ToDo: random saturation
        #pca whitening
#                tmpmean = np.mean(random_cropped[r,:,:,:])
#                tmpstd = np.mean(random_cropped[r,:,:,:]**2) - tmpmean * tmpmean
#                random_cropped[r,:,:,:] = (random_cropped[r,:,:,:] - tmpmean) / tmpstd

    inp_exc = random_cropped

    yield inp_exc, targets[indices], indices

def getpreds(X, Y, fn, dataset, batchsize=128):
    errs = 0
    accs = 0
    batches = 0
    preds = []
    for batch in iterate_minibatches(X, Y, batchsize, shuffle=False):
        inputs, targets, _ = batch
        err, acc, pred = fn(inputs, targets)
        errs += err
        accs += acc
        batches += 1
        preds.append(pred)
    preds = np.vstack(preds)
    if dataset=='test':
        print("Final results:")
    print("  {} loss:\t\t\t{:.6f}".format(dataset, errs / batches))
    print("  {} accuracy:\t\t{:.2f} %".format(dataset, accs / batches * 100))
    return preds

def getFeat(X, dim, pos_ratio, l, input_var):
    import theano, lasagne
    feat = lasagne.layers.get_output(l, deterministic=True)
    feat_fn = theano.function([input_var], feat)
    feats = []


    for batch in iterate_minibatches(X, 'test', dim, 128, pos_ratio):
        rst = batch
        inputs = rst[0]
        feats.append(feat_fn(inputs))
    feats = np.vstack(feats)
    return feats


def getwresults(Y_unlabel, knn_label):
    unlabel_num = Y_unlabel.shape[0]
    knn_bin = np.zeros((unlabel_num, 11))
    for j in np.arange(unlabel_num):
        knn_bin[j] = np.bincount(knn_label[j])

    knn_bin = knn_bin[:, :10]

    maxi = np.argmax(knn_bin, axis = 1)
    maxi = maxi.astype('int32')
    maxv = knn_bin[np.arange(unlabel_num), maxi]
    sorted_vi = np.argsort(maxv)
    maxv = maxv[sorted_vi]
    print('prediction of w')
    printacc(sorted_vi, maxi, Y_unlabel)
    return maxv, maxi

def histog(feat_train, Y_train):
    pidx = np.where(Y_train==1)[0]
    nidx = np.where(Y_train==0)[0]
    pfeat = feat_train[pidx]
    nfeat = feat_train[nidx]
    pfeat_mean = np.mean(pfeat, 0)
    nfeat_mean = np.mean(nfeat, 0)

    dist_tt = np.linalg.norm(pfeat - pfeat_mean, axis=1)
    dist_tu = np.linalg.norm(pfeat - nfeat_mean, axis=1)
    dist_uu = np.linalg.norm(nfeat - nfeat_mean, axis=1)
    dist_ut = np.linalg.norm(nfeat - pfeat_mean, axis=1)

    b = 200
    plt.figure()
    plt.title('margin=1')
    plt.subplot(221)
    plt.title('pos to pos_mean')
    plt.hist(dist_tt, bins=b)
    plt.subplot(222)
    plt.title('pos to neg_mean')
    plt.hist(dist_tu, bins=b)
    plt.subplot(223)
    plt.title('neg to neg_mean')
    plt.hist(dist_uu, bins=b)
    plt.subplot(224)
    plt.title('neg to pos_mean')
    plt.hist(dist_ut, bins=b)

def printacc(idx, f_u, Y_unlabel):
    l = f_u.shape[0]
    f_u = f_u.astype('int32')
    lsts = [(0.9,1), (0.7,1), (0.5,1), (0.4,1), (0.3,1), (0.2,1), (0.1,1),(0,1)]
    for lst in lsts:
        tidx = idx[np.int(l * lst[0]):np.int(l * lst[1])]
        print('confidence %.1f-%.1f acc: %.3f' % (lst[0], lst[1], np.mean(f_u[tidx] == Y_unlabel[tidx])))
#        bct = np.bincount(f_u[tidx])
#        print('      diversity ', bct)

def printacc2(idx, f_u, Y_unlabel):
    l = f_u.shape[0]
    lsts = [5000, 10000, 15000, 20000, 25000, 30000]
    for lst in lsts:
        tidx = idx[np.maximum((l-lst),0):l]
        print(np.mean(f_u[tidx] == Y_unlabel[tidx]))

def get_pseudolabel(Feat_label, Feat_unlabel, Y_label, Y_unlabel, K=200):
#    d_uu = cdist(Feat_unlabel, Feat_unlabel)
#    idx_uu = np.argsort(d_uu, axis = 1)
#    idx_uu = idx_uu[:, :K]
#    W_uu = np.zeros_like(d_uu)
#    tidx = np.tile(np.arange(W_uu.shape[0]), [K, 1]).T
#    W_uu[tidx, idx_uu] = 1
#    W_uu = W_uu / np.sum(W_uu, 1)[:, np.newaxis]
#    W_uu = (W_uu + W_uu.T)/2
#    D_uu = np.diag(np.sum(W_uu, 1))

    d_ul = cdist(Feat_unlabel, Feat_label)
    idx_ul = np.argsort(d_ul, axis = 1)

    unlabel_num = Y_unlabel.shape[0]
    knn_label = np.tile(Y_label, [unlabel_num, 1])
    for i in np.arange(unlabel_num):
        knn_label[i, idx_ul[i, K:]] = 10

    idx_ul = idx_ul[:, :K]
    W_ul = np.zeros_like(d_ul)
    tidx = np.tile(np.arange(W_ul.shape[0]), [K, 1]).T
    W_ul[tidx, idx_ul] = np.exp(-d_ul[tidx, idx_ul])
    W_ul = W_ul / np.sum(W_ul, 1)[:, np.newaxis]


    #    W_uu = np.exp(-W_uu)
    #    W_ul = np.exp(-W_ul)

    #f_u = np.dot(np.linalg.inv(D_uu - W_uu), np.dot(W_ul, Y_train))
    f_u = np.dot(W_ul,Y_label)
    return f_u, knn_label


def get_mtresults(X, dataset, train_or_test, fn, num_cls, dim, pos_ratio=0.5,
        shuffle=False, augment=False, batchsize=128, w=None):
    from sklearn.utils import linear_assignment_
    import time

    start_time = time.time()
    errs = 0
    accs = 0
    batches = 0

    preds = []
    ys = []

    if w is not None:
        assert w.shape[0]==X['boxes'].shape[0]

    for batch in iterate_minibatches(X, train_or_test, dim, batchsize, pos_ratio, shuffle, augment):
        inputs, targets, excerpt = batch
        if w is None:
            err, acc, pred = fn(inputs, targets)
        else:
            err, acc, pred = fn(inputs, targets, w[excerpt])
        # print(err)
        errs += err
        accs += acc
        batches += 1

        preds.append(pred)
        ys.append(targets)

    preds = np.vstack(preds)
    ys = np.concatenate(ys)

    assert(num_cls==preds.shape[1])
    yhat = np.argmax(preds, axis=1)
    acc_total = np.mean(yhat == ys)

    # Then we print the results for this epoch:
    print("  took {:.3f}s".format(time.time() - start_time))
    print("      {} loss:\t\t{:.6f}".format(dataset, errs / batches))
    print("      {} accuracy:\t\t{:.2f} %".format(dataset, acc_total * 100))


    return preds
