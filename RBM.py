#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#Inspired from https://gist.github.com/myme5261314/005ceac0483fc5a581cc
__author__ = 'Qinglin Dong'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np

lr = 0.001
epochs = 1
batchsize = 5
weight_decay_rate = 0.1
errors = []
global_step=tf.Variable(0)

sess = tf.Session()

def sample_bernoulli(probs):
    return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

def sample_gaussian(x, sigma=0.01):
    return x + tf.random_normal(tf.shape(x), mean=0.0, stddev=sigma, dtype=tf.float32)

class RBM(object):
    def __init__(self,visible_size, hidden_size):     
        self.vN = visible_size
        self.hN = hidden_size

        self.W = tf.Variable(np.zeros([self.vN, self.hN], np.float32))
        self.vb = tf.Variable(np.zeros([self.vN], np.float32))
        self.hb = tf.Variable(np.zeros([self.hN], np.float32))

    def build(self):
        self.X = tf.placeholder("float", [None, self.vN])

        _h0= tf.nn.sigmoid(tf.matmul(self.X, self.W) + self.hb)  #probabilities of the hidden units
        h0 = sample_bernoulli(_h0)
        _v1 = (tf.matmul(h0, tf.transpose(self.W)) + self.vb)
        v1 = sample_gaussian(_v1)
        h1 = tf.nn.sigmoid(tf.matmul(v1, self.W) + self.hb)

        w_pos_grad = tf.matmul(tf.transpose(self.X), h0)
        w_neg_grad = tf.matmul(tf.transpose(v1), h1)
        CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(self.X)[0])

        L1G = weight_decay_rate * (tf.sign(self.W))

        #alpha=tf.train.inverse_time_decay(lr,global_step,10,0.5)
        alpha=lr
        self.update_W = tf.assign_add(self.W, alpha * (CD-L1G))
        self.update_vb = tf.assign_add(self.vb, alpha * tf.reduce_mean(self.X - v1, 0))
        self.update_hb = tf.assign_add(self.hb, alpha * tf.reduce_mean(h0 - h1, 0))

        self.err = tf.reduce_mean(tf.square(self.X - v1))
        self.l1=tf.reduce_mean(tf.abs(self.W))

        sess.run(tf.global_variables_initializer())

    def train(self,data):

        weight_decays=[]
        G=[]
        M=[]


        for epoch in range(epochs):

            for start, end in zip( range(0, len(data), batchsize), range(batchsize, len(data), batchsize)):
                batch = data[start:end]
                #print start
                #print end
                sess.run([self.update_W, self.update_vb, self.update_hb],feed_dict={self.X: batch})
                if start % 10000 == 0:
                    errors.append(sess.run(self.err, feed_dict={self.X: batch}))
                    #weight_decays.append(sess.run(self.l1))
                    #G.append(sess.run(tf.reduce_mean(tf.abs(self.update_W)),feed_dict={self.X: batch}))
                    #M=(sess.run(tf.reduce_max(self.W)))
            #print 'Epoch: %d' % epoch, 'Error: %f' % errors[-1], 'W mean: %f' % weight_decays[-1], 'W Gradient %f' % G[-1], 'W Max %f' % M
            #print 'Epoch: %d' % epoch, 'Error: %f' % errors[-1]
    
#rbmup inspired by https://github.com/myme5261314/dbn_tf/blob/master/rbm_tf.py 
    def rbmup(self, trX):
        X = tf.placeholder("float", [None, self.vN])
        h1=(tf.matmul(X, self.W) + self.hb)
        a=sess.run(h1, feed_dict={ X: trX})
        import sklearn.preprocessing
        h1 = sklearn.preprocessing.normalize(a)
        return h1

def MNIST():
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("/home/uga_qinglin/Documents/github/myDBN/MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    rbm=RBM(trX.shape[1], 500)
    rbm.build()
    h1=rbm.train(trX)
    np.save('W1.npy',sess.run(rbm.W))

    from utils import tile_raster_images
    tile_raster_images(X=sess.run(rbm.W).T, img_shape=(28, 28), tile_shape=(25, 20), tile_spacing=(1, 1))
    import matplotlib.pyplot as plt
    from PIL import Image
    image = Image.fromarray(tile_raster_images(X=(sess.run(rbm.W)).T, img_shape=(28, 28) ,tile_shape=(25, 20), tile_spacing=(1, 1)))
    ### Plot image
    plt.rcParams['figure.figsize'] = (18.0, 18.0)
    imgplot = plt.imshow(image)
    imgplot.set_cmap('gray')
    plt.savefig('Weights1.png')


def MRI():
    import scipy.io as sio

    trX = np.load("/srv1/HCP_4mm/205725/MNINonLinear/Results/tfMRI_EMOTION_LR/tfMRI_EMOTION_LR_hp200_s4.feat/signal.npy")
    import sklearn.preprocessing
    trX=sklearn.preprocessing.scale(trX)

    rbm=RBM(trX.shape[1], 100)
    rbm.build()

    rbm.train(trX)
    np.save('W1.npy',sess.run(rbm.W))

    h1=rbm.rbmup(trX)
    np.save('h1.npy',h1)
    rbm=RBM(100, 100)
    rbm.build()
    rbm.train(h1)
    np.save('W2.npy',sess.run(rbm.W))

    h2=rbm.rbmup(h1)
    np.save('h2.npy',h2)
    rbm=RBM(100, 100)
    rbm.build()
    rbm.train(h2)
    np.save('W3.npy',sess.run(rbm.W))

    from subprocess import call
    call(["matlab","-r","run('Map.m');quit;"])

def MRI900():
    import scipy.io as sio
    
    
    #trX = np.load("/srv1/HCP_4mm/100307/MNINonLinear/Results/tfMRI_RELATIONAL_LR/tfMRI_RELATIONAL_LR_hp200_s4.feat/signal.npy")
    #rbm1 = RBM(trX.shape[1], 100)
    rbm1 = RBM(28549, 100)
    rbm2 = RBM(100, 100)
    rbm3 = RBM(100, 100)
    rbm1.build()
    rbm2.build()
    rbm3.build()

    from pathlib import Path
    from itertools import islice
    for ite in range(0,100):
        #for path in pathlist[:2]:
	pathlist = Path("/srv1/HCP_4mm/").glob('**/tfMRI_EMOTION_LR/**/signal.npy')
        gen = (path for path in pathlist)
        #for p in islice(gen,1,2):
        print 'Epoch: %d' % ite  # , 'Error: %f' % errors[-1]
        for p in islice(gen, 1,100):
            # because path is object not string
            path_in_str = str(p)
            #print(idx)
            print(path_in_str)
            trX=np.load(path_in_str)

            import sklearn.preprocessing
            trX=sklearn.preprocessing.scale(trX)

            from sklearn.utils import shuffle
            trX = shuffle(trX)

            rbm1.train(trX)
            h1 = rbm1.rbmup(trX)
            rbm2.train(h1)
            h2 = rbm2.rbmup(h1)
            rbm3.train(h2)
            h3 = rbm3.rbmup(h2)

    np.save('W1.npy', sess.run(rbm1.W))
    np.save('W2.npy', sess.run(rbm2.W))
    np.save('W3.npy', sess.run(rbm3.W))
    np.save('h1.npy', h1)
    np.save('h2.npy', h2)
    np.save('h2.npy', h3)

    from subprocess import call
    call(["matlab", "-r", "run('Map.m');quit;"])
if __name__ == '__main__':
    MRI900()
    #MNIST()
