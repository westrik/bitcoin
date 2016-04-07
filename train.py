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

import sys
import pickle
import numpy as np
import sklearn
import sklearn.cluster


def load(filename):
    """
    Load csv training file of form: unix_timestamp, price, num_bid, num_ask
    """
    csv = np.genfromtxt(\
            filename, \
            dtype=[('time','i8'),('price','f8'),('bid','f8'),('ask','f8')], \
            delimiter=",")

    if len(csv) == 0:
        print "Invalid training data"
        quit()

    return csv


def split_into_intervals(data, n):
    """
    Split time series into n minute intervals
    """
    # Throw away time, bid/ask numbers
    prices =  [x[1] for x in data]

    # create a len n-1 array of price differences (10 second increments)
    price_diffs = np.diff(prices)

    # m = interval length in terms of data points (6*~10sec = 1 minute)
    m = n * 6

    # each datapoint we're trying to cluster will be of the form:
    #     (xi,yi) = (time series of prices, price change after series)
    intervals = np.zeros((len(prices)-1,m+1))

    for i in range(0, len(prices)-m-1):
        intervals[i,0:m] = prices[i:i+m]
        intervals[i,m] = price_diffs[i+m]

    return intervals


def cluster(data):
    """
    Use k-means clustering on training data to find profitable patterns
    we can exploit
    """

    num_clusters = 100
    num_selected_clusters = 20

    # Split into 30, 60, and 120 min time intervals, cluster each
    split = lambda n: split_into_intervals(data, n)
    kmeans30 = sklearn.cluster.k_means(split(30), num_clusters)
    kmeans60 = sklearn.cluster.k_means(split(60), num_clusters)
    kmeans120 = sklearn.cluster.k_means(split(120), num_clusters)

    # Sort the clusters by performance
    hp30, hp60, hp120 = [], [], []
    for i in range(0, num_clusters):
        hp30.append((i,kmeans30[0][i,-1]))
        hp60.append((i,kmeans60[0][i,-1]))
        hp120.append((i,kmeans120[0][i,-1]))

    hp30 = sorted(hp30, reverse=True, key=lambda x: x[1])[0:num_selected_clusters]
    hp60 = sorted(hp60, reverse=True, key=lambda x: x[1])[0:num_selected_clusters]
    hp60 = sorted(hp120, reverse=True, key=lambda x: x[1])[0:num_selected_clusters]

    # Select the highest performing clusters
    top30 = np.zeros((num_selected_clusters,181))
    top60 = np.zeros((num_selected_clusters,361))
    top120 = np.zeros((num_selected_clusters,721))

    for i in range(0, num_selected_clusters):
        top30[i,0:181] = kmeans30[0][hp30[i][0],0:181]
        top60[i,0:361] = kmeans60[0][hp60[i][0],0:361]
        top120[i,0:721] = kmeans120[0][hp120[i][0],0:721]

    # Then normalize the clusters so we can use the faster similarity function
    #    from S&Z to compare instead of L2 norm
    scaler = sklearn.preprocessing.StandardScaler()
    for i in range(0, num_selected_clusters):
        top30[i,0:180] = scaler.fit_transform(top30[i,0:180])
        top60[i,0:360] = scaler.fit_transform(top60[i,0:360])
        top120[i,0:720] = scaler.fit_transform(top120[i,0:720])

    return [top30, top60, top120]


def similarity(a, b):
    """
    Calculate similarity metric (as defined by S&Z)

    s(a, b) = (Σ z=1→M (a_z - mean(a))(b_z - mean(b)))/(M*std(a)*std(b))
    """

    if len(a) != len(b):
	raise Exception("Vectors are not aligned")
    elif len(a) == len(b) == 0:
	raise Exception("Vectors are empty")


    numerator = np.sum((np.subtract(a, np.mean(a)))*(np.subtract(b, np.mean(b))))
    #numerator = 0
    #for z in range(0, len(a)):
    #    numerator += (a[z]-np.mean(a))*(b[z]-np.mean(b))
    denominator = len(a)*np.std(a)*np.std(b)

    if (denominator == 0):
       return numerator

    return numerator / denominator


def predict(prices, clusters):
    """
    Predict ∆p (change in price prior to interval) using Bayesian regression:

    ∆pⱼ = y • (exp(c(x,xᵢ))))/(Σ i=1→n (exp(c(x,xᵢ))))
    ∆pⱼ = Σ i=1→n (yᵢ * exp(c(x,xᵢ))))/(Σ i=1→n (exp(c(x,xᵢ))))
    """

    num_clusters = len(clusters)
    len_interval = len(prices)

    if len(prices) != len(clusters[0])-1:
        raise Exception ("Vector is wrong size: "+str(len_interval))

    # S&Z doesn't discuss how to select c, TODO experiment
    c, numerator, denominator = -1, 0, 0

    for i in range(0, num_clusters):
        distance = np.exp(c*similarity(prices, clusters[i][0:len_interval]))
        numerator += distance*clusters[i][-1]
        denominator += distance

    return(numerator / denominator)


def fit_weights(training_data, ys):
    # S&Z doesn't specify a lambda/alpha value, TODO experiment
    lasso = sklearn.linear_model.Lasso(alpha = 0.00000001)

    lasso.fit(training_data, ys)

    return lasso.intercept_ + lasso.coef_ 


def train(training_data, clusters):
    """
    Use B.regression with clustered data to predict new dataset
    Then use those predicted vals to fit our weights to:
        ∆p = w₀ + w₁∆p₁ + w₂∆p₂ + w₃∆p₃ + w₄r
    """

    vals = np.zeros((len(training_data)-721,4))
    results = np.zeros((len(training_data)-721,1))

    # Iterate through training data, at each point try to predict price
    seq = lambda n: [x[1] for x in training_data[i-n:i]]
    for i in range(720, len(training_data)-1):
        p1 = predict(seq(180), clusters[0])
        p2 = predict(seq(360), clusters[1])
        p3 = predict(seq(720), clusters[2])
        ask = training_data[i][2]
        bid = training_data[i][3]
        r = (bid-ask)/(bid+ask)
        deltap = training_data[i+1][1] - training_data[i][1]

        vals[i-720][0:4] = [p1,p2,p3,r]
        results[i-720][0] = deltap

    weights = fit_weights(vals, results)

    return weights


# Train the model
if __name__=="__main__":
    if (len(sys.argv)) == 1:
        print "Need csv with training data"
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
    pickle.dump(clusters, open("weights/clusters.pickle", "wb"))
    pickle.dump(weights,  open("weights/weights.pickle", "wb"))
