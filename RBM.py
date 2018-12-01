
# Inspired from https://gist.github.com/myme5261314/005ceac0483fc5a581cc
#__author__ = 'Qinglin Dong'

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import sklearn.preprocessing

def sample_bernoulli(probs):
    return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

def sample_gaussian(x, sigma=0.01):
    return x + tf.random_normal(tf.shape(x), mean=0.0, stddev=sigma, dtype=tf.float32)

class RBM():


    def __init__(self, visible_size, hidden_size,
                 lr = 0.01, epochs = 20,
                 batchsize = 20, weight_decay_rate = 0.1):
        self.vN = visible_size
        self.hN = hidden_size

        self.lr = lr
        self.epochs = epochs
        self.batchsize = batchsize
        self.weight_decay_rate = weight_decay_rate

        self.errors=[]
        self.X = tf.placeholder("float", [None, self.vN])

        self.W = tf.Variable(np.zeros([self.vN, self.hN], np.float32))
        self.vb = tf.Variable(np.zeros([self.vN], np.float32))
        self.hb = tf.Variable(np.zeros([self.hN], np.float32))

        self._h0 = tf.nn.sigmoid(tf.matmul(self.X, self.W) + self.hb)  # probabilities of the hidden units
        self.h0 = sample_bernoulli(self._h0)
        self._v1 = (tf.matmul(self.h0, tf.transpose(self.W)) + self.vb)
        self.v1 = sample_gaussian(self._v1)
        self.h1 = tf.nn.sigmoid(tf.matmul(self.v1, self.W) + self.hb)

        self.w_pos_grad = tf.matmul(tf.transpose(self.X), self.h0)
        self.w_neg_grad = tf.matmul(tf.transpose(self.v1), self.h1)

        self.CD = (self.w_pos_grad - self.w_neg_grad) / tf.to_float(tf.shape(self.X)[0])
        self.L1G = self.weight_decay_rate * (tf.sign(self.W))

        self.update_W = tf.assign_add(self.W, self.lr * (self.CD - self.L1G))
        self.update_vb = tf.assign_add(self.vb, self.lr * tf.reduce_mean(self.X - self.v1, 0))
        self.update_hb = tf.assign_add(self.hb, self.lr * tf.reduce_mean(self.h0 - self.h1, 0))

        self.err = tf.reduce_mean(tf.square(self.X - self.v1))
        self.l1 = tf.reduce_mean(tf.abs(self.W))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, data):
        for epoch in range(self.epochs):            
            for start, end in zip(range(0, len(data), self.batchsize), range(self.batchsize, len(data), self.batchsize)):
                
                batch = data[start:end]
                self.sess.run([self.update_W, self.update_vb, self.update_hb], feed_dict={self.X: batch})
            
                if start % 1000 == 0:
                    print("Epoch: "+ str(epoch) + " Batch: "+ str(start)+ "-" +str(end)+
                         " Error: "+ str(self.sess.run(self.err, feed_dict={self.X: batch})) + 
                         " l1: "+str(self.sess.run(self.l1))) 

                    # rbmup inspired by https://github.com/myme5261314/dbn_tf/blob/master/rbm_tf.py
    def predict(self, data):
        h1 = (tf.matmul(self.X, self.W) + self.hb)
        a = self.sess.run(h1, feed_dict={self.X: data})
        h1 = sklearn.preprocessing.normalize(a)
        return h1

    def getW(self):
        W=self.sess.run(self.W)
        return W