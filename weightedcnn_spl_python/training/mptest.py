# -*- coding: utf-8 -*-

from  multiprocessing import Pool
import sys
sys.path.append('/media/dl/data2/cifar10_resnet/') 
from utils import *

def gg(aa, bb):
    import theano
    import theano.tensor as T
    a = T.scalar(name='a', dtype='float32')
    b = T.scalar(name='b', dtype='float32')
    c = a+b
    f = theano.function([a,b], c)
    return f(aa,bb)
if __name__ == '__main__':
    pool = Pool(processes=2)
    pool.apply_async(gg, (1.0,2.0))
    #r = pool.apply_async(train_cls, (data, cls_target_var, cls_train_fn, cls_val_fn))
    
    #kl_label = pool.apply_async(model_kl, (data,), kwargs)
    pool.close()
    pool.join()