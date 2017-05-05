# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import numpy as np

#x,y=T.matrices('xy')
#
## regular softmax and crossentropy
##sm = T.nnet.softmax(x)
#
##The problem goes away if you explicitly write out the softmax function instead of using Theano's:
#sm = T.exp(x)/(T.exp(x).sum(1,keepdims=True))
#
#cm1=T.nnet.categorical_crossentropy(sm,y)
#g1 = T.grad(cm1.mean(),x)
#
## numerically stable log-softmax with crossentropy
#xdev = x-x.max(1,keepdims=True)
#lsm = xdev - T.log(T.sum(T.exp(xdev),axis=1,keepdims=True))
#sm2 = T.exp(lsm) # just used to show equivalence with sm
#cm2=-T.sum(y*lsm,axis=1)
#g2 = T.grad(cm2.mean(),x)
#
#
## create some inputs into a softmax that are large and labels
#a=np.exp(10*np.random.rand(5,10).astype(theano.config.floatX))
## create some one-hot coded labels
#b=np.eye(5,10).astype(theano.config.floatX)
################################################################################
#There is an optimization that stabilizes the expression, 
#but I think it gets triggered only if the target y is expressed as a vector of indices, not as a matrix of one-hot vectors.
x=T.matrix('x')
y=T.ivector('y')

# regular softmax and crossentropy
sm = T.nnet.softmax(x)
cm1=T.nnet.categorical_crossentropy(sm,y)
g1 = T.grad(cm1.mean(),x)

# numerically stable log-softmax with crossentropy
xdev = x-x.max(1,keepdims=True)
lsm = xdev - T.log(T.sum(T.exp(xdev),axis=1,keepdims=True))
sm2 = T.exp(lsm) # just used to show equivalence with sm
cm2=-lsm[T.arange(y.shape[0]), y]
g2 = T.grad(cm2.mean(),x)


# create some inputs into a softmax that are large and labels
a=np.exp(10*np.random.rand(5,10).astype(theano.config.floatX))
# create some one-hot coded labels
b=np.random.randint(0,10,5).astype(np.uint8)
##############################################################################
# show equivalence of softmax and exponentiated numerically stable log-softmax
f1=theano.function([x],[sm, sm2])
sm1,sm2=f1(a)
print np.allclose(sm1,sm2)

# now show that the two versions result in the same crossentropy cost
# this indicates that the forward function does provide some numerical stability
f2=theano.function([x,y],[cm1,cm2])
c1,c2 = f2(a,b)
print np.allclose(c1,c2)

# now, show that in the standard softmax case the gradients blow up 
# while in the log-softmax case they don't
f3=theano.function([x,y],[g1,g2])
g1_,g2_ = f3(a,b)
print g1_
print g2_