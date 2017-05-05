# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import numpy as np
from theano import OrderedUpdates
a = T.ivector('a')
b = theano.shared(np.array([1,2]), 'int32')
loss = 0
c = theano.shared(1)
for i in np.arange(2):
    bn = T.set_subtensor(b[i], b[i]**2)
    loss += b[i] * a[i]
f = theano.function([a], [b, c,loss], updates=((c,c+1), (b,bn)))
av = np.array([1,2]).astype('int32')
print f(av)
print f(av)

#a = theano.shared(np.array([[1,2], [3,4]], dtype='float32'))
#b = T.fscalar('b')
#def loss(i, p):
#    an = T.set_subtensor(a[i], a[i]**2)
#    p += T.sum(an[i])
#    #return (p, OrderedUpdates([(a, an)]))
#    return (p, {a: an})
#
#results, updates = theano.scan(fn=loss, sequences=T.arange(2), outputs_info=b)
#fun = theano.function([b], [a, results[-1]], updates=updates)
#print fun(1)
#print fun(1)

#a = theano.shared(np.array([[1,2], [3,4]], dtype='float32'))
#b = T.fscalar('b')
#def loss(i, p, a):
#    an = T.set_subtensor(a[i], a[i]**2)
#    p += T.sum(an[i])
#    #return (p, OrderedUpdates([(a, an)]))
#    return (p, an)
#
#results, updates = theano.scan(fn=loss, sequences=T.arange(2), outputs_info=[b, a])
#p = results[0][-1]
#updates = {a: results[1][-1]}
#fun = theano.function([b], [a, p], updates=updates)
#print fun(1)
#print fun(1)

#a = theano.shared(np.array([[1,2], [3,4]], dtype='float32'))
#b = T.fscalar('b')
#def loss(i, b):
#    ai = a[i]
#    ai = ai ** 2
#    an = T.set_subtensor(a[i], a[i]**2)
#    b += T.sum(an[i])
#    u.update({a[i]: an[i]})
#    #return (p, OrderedUpdates([(a, an)]))
#    return (b, u)
#
#u = OrderedUpdates()
#results, updates = theano.scan(fn=loss, sequences=T.arange(2), outputs_info=[b, u])
#fun = theano.function([b], [a, results[-1]], updates=updates)
#print fun(1)
#print fun(1)

