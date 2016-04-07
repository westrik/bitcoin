from train import *

def test_trading(data):
    # iterate over test data,
    # making decisions at each time step
    seq = lambda n: [x[1] for x in data[i-n:i]]
    for i in range(720, len(data)-1):
        p1 = predict(seq(180), clusters[0])
        p2 = predict(seq(360), clusters[1])
        p3 = predict(seq(720), clusters[2])
        ask = data[i][2]
        bid = data[i][3]
        r = (bid-ask)/(bid+ask)

        pred = weights[0] + weights[1]*p1 + weights[2]*p2 + weights[3]*p3 + weights[4]*r

	print "prediction"+str(pred)

# Test the model
if __name__=="__main__":
    if (len(sys.argv)) == 1:
        print "Need csv with testing data"
        quit()

    # load dataset
    data = load(sys.argv[1])

    # load clusters and weights
    try:
        clusters = pickle.load(open("clusters.pickle", "rb"))
        weights = pickle.load(open("weights.pickle", "rb"))
    except:
        print "Generate weights and clusters with train.py"
        quit()
