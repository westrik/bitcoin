# -*- coding: utf-8 -*-

"""

Bitcoin trading bot
    Predict Bitcoin price changes using Bayesian regression 
    Implementation closely follows http://arxiv.org/abs/1410.1231

-------------------

Training and usage:

* Parse csv
* Split data into three groups:

* First group:
   Create subgroups of 30/60/120 minutes, along with price change immediately 
    following each interval (10 sec)
   Cluster with k-means (100 groups)
   Pick 20 highest performing clusters

* Second group:
   Create subgroups of 30/60/120 minutes
   Use Bayesian regression with 20 clusters to predict price change for 
       intervals in second group
   Fit weights from ∆p model function to predicted data

* Third group:
   Test fitted ∆p model:
   For a given data point, estimate ∆p for 30/60/120 minutes prior to data 
       point, plug estimations into ∆p model. Compare ∆p to threshold to make 
       trade decision

"""

import csv
import sys

import numpy as np
import sklearn
from sklearn.cluster import KMeans

def load(filename):
    csv = np.genfromtxt(\
            filename, \
            dtype=[('time','i8'),('price','f8'),('bid','f8'),('ask','f8')], \
            delimiter=",")

    if len(csv) == 0:
        print "Invalid training data"
        quit()

    return csv


def split_into_intervals(data):
    pass


def cluster(data):
    # split into 30, 60, and 120 min time intervals
    pass
    #sklearn.cluster.k_means(data,100)


def train(training_data, clusters):
    pass


if __name__=="__main__":
    if (len(sys.argv)) == 1:
        print "Need csv"
        quit()

    # load dataset
    data = load(sys.argv[1])

    # split dataset into 2, skipping every other element
    # i.e. turn 5s increment into 10s increment
    cluster_data = data[:len(data)/2][::2]
    train_data = data[len(data)/2:][::2]

    # cluster the first part of data
    clusters = cluster(cluster_data)

    # fit params using second part of data
    weights = train(train_data, clusters)

    # save weights and clusters for later usage
    # ... TODO
