filename = 'data.csv';
M = csvread(filename);

labels = M(:,1);

M(:,1) = [];

D = pdist(M);


% func return label

% func return distance between two cars lp

% func get L, and solves |z_i - z_j| < L*d(c1,c2_

% func get L, and solves |z_i - z_j| < L*d(c1,c2_