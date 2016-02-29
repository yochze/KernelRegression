import time
import math
import scipy
import numpy
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linprog # linear programming model 

class Nadarya:

    """
    Compute the Nadarya-Watson Estimator
    """


    def __init__(self, data_csv, limit=10, train_ratio=0.5, H=10):
        m = numpy.loadtxt(data_csv, delimiter=",")

        # Load data
        self.H = H 
        labels = []
        vectors = []

        # Generate seperate labels and vectors arrays
        for vector in m[:limit]:
            labels.append(vector[0])
            vectors.append((vector[1:]))

        train_size = int(math.ceil(limit * train_ratio))
        test_size  = limit - train_size

        self.train = vectors[:train_size]
        self.test  = vectors[train_size:]

        self.train_labels = labels[:train_size]
        self.test_labels  = labels[train_size:]


    def kernel(self,x,x_i,H):
        """
           K_h_x = a*e^(-d(p1,p2)/h^2)
        """
        a = 1 
        k_i = a * math.exp(-1 * (distance.euclidean(x,x_i)) / (H ** 2)) 

        return k_i

    def compute(self,x, xs, ys, H):
        """
        sum ( K_h_x * ( x - x_i ) * y_i )
           /
       sum ( K_h_x * (x - x_i ) )

        """
        nom_sum = 0
        denom_sum = 0
        for i,x_i in enumerate(xs):
            k = self.kernel(x, x_i, H)
            nom_sum   += k * distance.euclidean( x , x_i ) * ys[i]
            denom_sum += k * distance.euclidean( x , x_i )


        return (nom_sum / denom_sum)

    def compare_results(self, hyp,ref):
        ec = len(hyp) # elements count
        error_sum = 0
        zipped = zip(ref, hyp)

        for i,z in enumerate(hyp):
            error_sum += abs(z - ref[i])
        return error_sum / ec 


def main(H, data_size, train_ratio):
    """
        Load data
    """
    nadarya = Nadarya('data.csv', data_size, train_ratio, H)

    hyp_res = []
    xs_range = int(data_size * train_ratio)

    start = time.time()
    for point in nadarya.test:
        hyp_res.append(nadarya.compute(point, nadarya.train, nadarya.train_labels, H ))

    diff = nadarya.compare_results(hyp_res, nadarya.test_labels)
    duration = time.time() - start

    return [diff, duration]


if __name__ == "__main__":
    H = [1, 10, 100, 1000]
    print("H\tscore\tduration\ttrain_ratio\tdata_size")
    for h in H:
        data_size = 200
        train_ratio = 0.5
        diff, duration = main(h, data_size, train_ratio)
        res = "{}\t{}\t{}\t{}\t{}".format(h, diff, duration, train_ratio, data_size)
        print(res)
