import csv
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets

if (len(sys.argv)) == 1:
    print "Need csv"
    quit()

#
#with open(sys.argv[1], 'rb') as data:
#    d = csv.reader(data, delimiter=' ', quotechar='|')
#    for row in d:
#        print ', '.join(row)





# Bayesian inference


# delta p = w_0 + sum (j=1->3) w_j delta p^j + w_4 r

# w = (w0, . . . , w4) are learned parameters




# similarity
# s(a, b) = sum (z=1->M)(a_z -mean(a))(b_z-mean(b)) / 
#               M std(a) std(b)

#        mean(a) = (sum (z=1->M) a_z)/M
#        std(a) = (sum (z=1->M) a_z - mean(a)^2)/M
