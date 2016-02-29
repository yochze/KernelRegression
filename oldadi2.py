import scipy
import numpy
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linprog # linear programming model


class ADI:

    def __init__(data_csv='data.csv', limit=100, train_ratio=0.5, L=0.1):
        self.m = numpy.loadtxt('data.csv', delimiter=",")
        
        # Load data
        self.labels, self.vectors = [], []
        self.L = L
        
        # Generate seperate labels and vectors arrays
        for vector in m[:10]:
          labels.append(vector[0])
          vectors.append((vector[1:]))


        train_size = math.ceil(limit * train_ratio)
        test_size  = limit - train_size

        self.train = vectors[:train_size]
        self.test  = vectors[train_size:]
        self.train_labels = labels[:train_size]
        self.test_labels  = labels[train_size:]


    
    def dist_matrix(vectors, dfunction):
        # Create the distance matrix (NxN), symmetric matrix
        dist_matrix = squareform(pdist(vectors, 'euclidean'))


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
        ub_ij = L * dist_matrix.item( (i, j) )
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
    ub_qs = [ -1 * labels[i], labels[i] ]
    ub.extend(ub_qs)
# print("A")
# print(A)
# print("Upper bounds")
# print(ub)

# Linear programming
res = linprog(F, A_ub=A, b_ub=ub, bounds=None, options={"disp": True})
# print(res)

zs = res.slack
# Phase 2

# Given a new POINT, calculate its tagging by
# for each pair (z_i, z_j)

def find_best_z(new_point, points, lp_slack):
    """
    Given a new point, calculate its best tagging through constructing matrix
    Z that for for each z_i z_j (p_i, p_j tags) calculate z_k such that:
    z_k  = (z_i * dist(p_i, p_j) + z_j * dist(p_j,p_k)) / (dist(p_i, p_j) + dist(p_2,p_3))
    """
    Z = []

    for i,z_i in enumerate(lp_slack):
        Z.append( [0] * len(lp_slack))
        for j,z_j in enumerate(lp_slack):
            
            z_k = ((z_i * distance.euclidean(points[i], points[j])) + (z_j * distance.euclidean(points[j], new_point))) / (distance.euclidean(points[i],points[j]) + distance.euclidean(points[j], new_point))

            Z[i][j] = z_k

    return numpy.argmin(Z)

print(find_best_z(new_point,vectors, zs[:len(vectors)-1]))

