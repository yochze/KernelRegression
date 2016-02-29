import scipy
import numpy
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linprog # linear programming model


# Load data file
m = numpy.loadtxt('data.csv', delimiter=",")
labels  = []
vectors = []

# Parameters 
L = 10 # distance multiplier
F = [0, 0, 1]  # objective function

# Generate seperate labels and vectors arrays
for vector in m:
  labels.append(vector[0])
  vectors.append((vector[1:]))

# We have 391 vectors
# Create the distance matrix (391x391), symmetric matrix
dist_matrix = squareform(pdist(vectors, 'euclidean'))


# Generate the A matrix (based on q and z constraints):
# (1) |z_i-z_j| <= L*Dist(c_i, c_j)  - for each pair
# and
# (2) |label(c_i) - z_i| <= q_i - for each vector

# For each vector we'll build the A coefficients matrix
ub = [] # Upper bound values
A  = [] # Coefficient matrix

for i,vi in enumerate(vectors):
    for j,vj in enumerate(vectors):
        # First constaint
        # Array positions is: z_i, z_j, q_i
        A.append( [1, -1, 0])
        A.append([-1, 1, 0])
        
        # Upper bounds
        ub_i = L * dist_matrix.item( (i, j) )
        ub.extend([ub_i, ub_i]) 


    # Second constrain, per vector
    A.append([-1, 0, -1])
    A.append([1, 0, -1])
    
    # Upper bounds
    ub_qs = [ -1 * labels[i], labels[i] ]
    ub.extend(ub_qs)

# Linear programming
res = linprog(F, A_ub=A, b_ub=ub, bounds=None, options={"disp": True})
print(res)
