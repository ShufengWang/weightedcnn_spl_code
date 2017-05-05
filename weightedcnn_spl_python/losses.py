# -*- coding: utf-8 -*-
import numpy as np

def intra_cls_loss(feat, target_var, cls_num, cls_feat_mean_avg=None):
    import theano.tensor as T
    mloss = T.constant(0.0)
    cls_feat_mean = {}
    cls_feat_mean_avg_new = cls_feat_mean_avg
    for i in np.arange(cls_num):
        cls_idx = T.eq(target_var, i).nonzero()[0]
        cls_feat = feat[cls_idx, :]
        cls_feat_mean[i] = T.mean(cls_feat, 0)
        cls_feat_mean_avg_new = T.set_subtensor(cls_feat_mean_avg_new[i], 
            cls_feat_mean_avg_new[i] - 0.5 * T.sum((cls_feat_mean_avg_new[i] - cls_feat), 0)/(1+cls_idx.shape[0]))
        #mean_updates.append((cls_feat_mean_avg[i], cls_feat_mean_avg_new))
        mloss += T.sum((cls_feat - cls_feat_mean_avg_new[i])**2) / feat.shape[0]
    
    
    return mloss, cls_feat_mean_avg_new
#    cls_feat_mean_avg = []
#    for i in np.arange(cls_num):
#        cls_feat_mean_avg.append(theano.shared(np.zeros((64,), dtype='float32')))
#        
#    mloss = []
#    for i in np.arange(cls_num):
#        cls_idx = T.eq(target_var, i).nonzero()[0]
#        cls_feat = feat[cls_idx, :]
#        cls_feat_mean = T.mean(cls_feat, 0)
#        mloss.append(T.sum((cls_feat - cls_feat_mean_avg[i])**2)/128)
#        cls_feat_mean_avg[i] = cls_feat_mean_avg[i] - 0.5 * T.sum((cls_feat_mean - cls_feat), 0)/(1+cls_idx.shape[0])
#    mloss = T.sum(mloss)
#    loss += 1 * mloss
#    
#    def clsi_loss(i, prior_loss, cls_feat_mean_avg, target_var, feat):
#        cls_idx = T.eq(target_var, i).nonzero()[0]
#        cls_feat = feat[cls_idx, :]
#        cls_feat_mean = T.mean(cls_feat, 0)
#        cls_feat_mean_avg_new = T.set_subtensor(cls_feat_mean_avg[i],
#            cls_feat_mean_avg[i] - T.sum((cls_feat_mean - cls_feat), 0)/(1+cls_idx.shape[0]))
#        iavg = cls_feat_mean_avg_new[i]
#        mloss = prior_loss + T.sum((cls_feat - iavg)**2) / 128
#        return (mloss, cls_feat_mean_avg_new)
#    results, mean_updates = theano.scan(fn=clsi_loss, sequences=T.arange(10), 
#                                      outputs_info = [0.0, cls_feat_mean_avg], 
#                                        non_sequences=[target_var, feat])
#    mloss = results[0][-1]
#    loss += mloss
#    mean_updates.update([(cls_feat_mean_avg, results[1][-1])])
#    
#    def clsi_loss(i, prior_loss, cls_feat_mean_avg, target_var, feat):
#        cls_idx = T.eq(target_var, i).nonzero()[0]
#        cls_feat = feat[cls_idx, :]
#        cls_feat_mean = T.mean(cls_feat, 0)
#        cls_feat_mean_avg_new = T.set_subtensor(cls_feat_mean_avg[i],
#            cls_feat_mean_avg[i] - T.sum((cls_feat_mean - cls_feat), 0)/(1+cls_idx.shape[0]))
#        iavg = cls_feat_mean_avg_new[i]
#        mloss = prior_loss + T.sum((cls_feat - iavg)**2) / 128
#        return (mloss, {cls_feat_mean_avg: cls_feat_mean_avg_new})
#    results, mean_updates = theano.scan(fn=clsi_loss, sequences=T.arange(10), 
#                                      outputs_info = 0.0, non_sequences=[cls_feat_mean_avg, target_var, feat])
#    mloss = results[-1]
#    loss += mloss

    
def get_klloss(n, margin, prediction, target_var):
    import theano.tensor as T
    log_prediction = T.log(prediction)
    prediction = prediction.reshape((n, 1, -1))
    logpred1 = log_prediction.reshape((n, 1, -1))
    logpred2 = log_prediction.reshape((1, n ,-1))
    logpred_m = logpred1 - logpred2
    kl = prediction * logpred_m
    kl = T.sum(kl, 2)

    id1 = target_var.reshape((-1,1))
    id2 = target_var.reshape((1,-1))
    sim = (id1-id2)
    idxes = T.neq(sim, 0).nonzero()
    kl = T.set_subtensor(kl[idxes], T.maximum(margin-kl[idxes],0))   
    return T.mean(kl)