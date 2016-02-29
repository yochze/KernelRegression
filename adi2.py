import time
import math
import scipy
import numpy
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linprog # linear programming model

import pdb # debug

class Adi:

    def __init__(self,data_csv='data.csv', limit=10, train_ratio=0.5, L=0.1, distance_function='euclidean'):

        m = numpy.loadtxt(data_csv, delimiter=",")

        # Load data
        self.L = L
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

        self.dist_matrix = self.create_dist_matrix(self.train, distance_function)


    def create_dist_matrix(self, vectors, dfunction):
        # Create the distance matrix (NxN), symmetric matrix
        return squareform(pdist(vectors, dfunction))

    def lin_prog(self, vectors, dist_matrix):
        # Generate the A matrix (based on q and z constraints):
        # (1) |z_i-z_j| <= L*Dist(c_i, c_j)  - for each pair
        # and
        # (2) |label(c_i) - z_i| <= q_i - for each vector

        # For each vector we'll build the A coefficients matrix
        ub = [] # Upper bound values
        A  = [] # Coefficient matrix

        C = (len(vectors) * 2) # the constraints coefficients vector size

        F = ([0] * len(vectors)) + ([1] * len(vectors))  # objective function (\sumQ_i)

        for i,vi in enumerate(vectors):
            for j,vj in enumerate(vectors):

                # (1) constaint
                ineq_1 = [0] * C
                ineq_2 = [0] * C

                if i != j:
                    ineq_1[i], ineq_1[j] = 1, -1
                    ineq_2[i] , ineq_2[j] = -1, 1


                # Append to A
                A.append( ineq_1 )
                A.append( ineq_2 )

                # Upper bounds
                ub_ij = self.L * dist_matrix.item( (i, j) )
                ub.extend([ub_ij, ub_ij]) 

            # Second constrain (Q), per vector
            ineq_1b = [0] * C
            ineq_1b[i] = -1 # Z
            ineq_1b[i + (len(vectors))] = -1 #Q_1

            ineq_1c = [0] * C
            ineq_1c[i] = 1 # Z
            ineq_1c[i + (len(vectors))] = -1 #Q_1


            # Append to A
            A.append(ineq_1b)
            A.append(ineq_1c)

            # Upper bounds
            ub_qs = [ -1 * self.train_labels[i], self.train_labels[i] ]
            ub.extend(ub_qs)


        # Linear programming
        res = linprog(F, A_ub=A, b_ub=ub, bounds=None, options={"disp": True})
        return(res)

    # Phase 2

    # Given a new POINT, calculate its tagging by
    # for each pair (z_i, z_j)

    def find_best_z(self, new_point, points, lp_x):
        """
        Given a new point, calculate its best tagging through constructing matrix
        Z that for for each z_i z_j (p_i, p_j tags) calculate z_k such that:
        z_k  = (z_i * dist(p_i, p_j) + z_j * dist(p_j,p_k)) / (dist(p_i, p_j) + dist(p_2,p_3))
        """

        """ V2 
            Get min max of
        """


        Z = []

        max_jump = 0
        best_z   = 0


        for i,z_i in enumerate(lp_x):
            for j,z_j in enumerate(lp_x):

                if z_i > z_j:
                    z1 = z_i
                    z2 = z_j
                    point1 = points[i]
                    point2 = points[j]

                else:
                    z1 = z_j
                    z2 = z_i
                    point1 = points[j]
                    point2 = points[i]


                z_k = ((z1 * distance.euclidean(point2, new_point)) + (z2 * distance.euclidean(point1, new_point))) / (distance.euclidean(point1,new_point) + distance.euclidean(point2, new_point))

                # Z[i][j] = z_k

                #  IF the jump (z1-z3 / dist(p1,p3)) is bigger then save z3

                if distance.euclidean(point1,new_point) == 0:
                    current_jump = 0
                else:
                    current_jump = ((z1 - z_k) / distance.euclidean(point1, new_point))

                if (current_jump > max_jump):
                    max_jump = current_jump
                    best_z = z_k

            return best_z


    def compare_results(self, hyp,ref):
        ec = len(hyp) # elements count
        error_sum = 0

        zipped = zip(ref, hyp)

        #print(list(zipped))


        for i,z in enumerate(hyp):
            error_sum += abs(z - ref[i])

        return (error_sum / ec) 

def main(L, distance_function, data_size, train_ratio):
    """
        Load data
    """
    adi = Adi('data.csv', data_size, train_ratio, L)
    dist_matrix = adi.dist_matrix

    # print(dist_matrix)

    lprog = adi.lin_prog(adi.train, dist_matrix)

    hyp_res = []
    xs_range = int(data_size * train_ratio)
    X = lprog.x[:xs_range]

    start = time.time()
    for point in adi.test:
        hyp_res.append(adi.find_best_z(point, adi.train, X))

    diff = adi.compare_results(hyp_res, adi.test_labels)
    duration = time.time() - start

    return [diff, duration]

if __name__ == "__main__":
    lipschitz = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    print("L\tscore\tduration\tdistance_function\ttrain_ratio\tdata_size")
    for l in lipschitz:
        data_size = 200
        train_ratio = 0.5
        diff, duration = main(l, 'euclidean', data_size, train_ratio)
        res = "{}\t{}\t{}\t{}\t{}\t{}".format(l, diff, duration, 'euclidean',  train_ratio, data_size)
        print(res)
