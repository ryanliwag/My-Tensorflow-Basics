#!/usr/bin/python3

'''
Made by: Ryan Joshua Liwag
'''

from __future__ import print_function, division

import json
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

import sys, argparse

logs_path = 'logs'

# Rerturns error rate between prediction and Y
def error_rate(p, t):
  return np.mean(p != t)

#Function to one_hot the target classes
def one_hot(data):
	S = pd.Series(list(data))
	one_hots = pd.get_dummies(S)
	return one_hots

#Function to load the IRIS dataset
def normalize():
	data = load_iris()
	X_data = data.data
	Y_data = data.target
	num_classes = len(data.target_names)
	df = pd.DataFrame(X_data)
	me = df.mean(axis = 0)
	std = df.std(axis = 0)
	print(std)
	X = (df - me) / std
	Y = Y_data
	return X, Y, num_classes

#def import_data(train, test):




class TFLogistic:
  def __init__(self, savefile, D=None, K=None):
    self.savefile = savefile
    if D and K:
      # we can define some parts in the model to be able to make predictions
      self.build(D, K)

  def build(self, D, K, nb_classes):


    W0 = np.random.randn(D, K) * np.sqrt(2.0 / D)
    b0 = np.zeros(K)

    # define variables and expressions
    self.inputs = tf.placeholder(tf.float32, shape=(None, D), name='inputs')
    self.targets = tf.placeholder(tf.int32, shape=(None,), name='targets')
    self.W = tf.Variable(W0.astype(np.float32), name='W')
    self.b = tf.Variable(b0.astype(np.float32), name='b')

    self.saver = tf.train.Saver({'W': self.W, 'b': self.b})

    with tf.name_scope('Model'):
    	logits = tf.matmul(self.inputs, self.W) + self.b

    with tf.name_scope('Loss'):
	    cost = tf.reduce_mean(
	        tf.nn.sparse_softmax_cross_entropy_with_logits(
	            logits=logits,
	            labels=self.targets
	        )
	    )

    self.predict_op = tf.argmax(logits, 1)

    tf.summary.scalar("Loss", cost)
    merged_summary_op = tf.summary.merge_all()
    
    return cost


  def fit(self, X, Y, Xtest, Ytest, nb_classes):
    N, D = X.shape

    K = len(Y)

    # hyperparams
    # Change and tweak this for training
    max_iter = 20000
    lr = 1e-3
    mu = 0.8
    regularization = 1e-1
    batch_sz = 50
    n_batches = N // batch_sz

    cost = self.build(D, K, nb_classes)
    l2_penalty = regularization*tf.reduce_mean(self.W**2) / 2
    cost += l2_penalty
    train_op = tf.train.MomentumOptimizer(lr, momentum=mu).minimize(cost)

    costs = []
    error = []
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        for i in range(max_iter):
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz),]
                Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz),]

                session.run(train_op, feed_dict={self.inputs: Xbatch, self.targets: Ybatch})

                test_cost = session.run(cost, feed_dict={self.inputs: X, self.targets: Y})

                Ptest = session.run(self.predict_op, feed_dict={self.inputs: X})
                err = error_rate(Ptest, Y)

                print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
                costs.append(test_cost)
                error.append(err)

        # save the model
        self.saver.save(session, self.savefile)

    # save dimensions for later
    self.D = D
    self.K = K

    plt.plot(costs)
    plt.show()
    plt.plot(error)
    plt.show()


  def predict(self, X):
    with tf.Session() as session:
      # restore the model
      self.saver.restore(session, self.savefile)
      P = session.run(self.predict_op, feed_dict={self.inputs: X})
    return P


  def score(self, X, Y):
    return 1 - error_rate(self.predict(X), Y)

  def save(self, filename):
    j = {
      'D': self.D,
      'K': self.K,
      'model': self.savefile
    }
    with open(filename, 'w') as f:
      json.dump(j, f)

  @staticmethod
  def load(filename):
    with open(filename) as f:
      j = json.load(f)
    return TFLogistic(j['model'], j['D'], j['K'])


def arguments(args=None):
	parser = argparse.ArgumentParser(description='Training script for logistic regression')
	parser.add_argument('-x', '--x_data',
						help = 'training inputs',
						required = 'True')
	parser.add_argument('-y', '--y_data',
						help = 'target inputs',
						required = 'True')
	args = parser.parse_args(args)
	return (args.x_data,
			args.y_data)






def main():

	#x_data, y_data = arguments()

    X, Y, nb_classes = normalize()
    X = X.values
    print(Y)
    Xtrain = X[:150]
    Ytrain = Y[:150]
    Xtest = X[100:]
    Ytest = Y[100:]

    model.save("my_trained_model.json")

    model = TFLogistic("tf.model")
    model.fit(Xtrain, Ytrain, Xtest, Ytest, nb_classes)

    # test out restoring the model via the predict function
    print("final train accuracy:", model.score(Xtrain, Ytrain))
    print("final test accuracy:", model.score(Xtest, Ytest))
  

if __name__ == '__main__':
    main()