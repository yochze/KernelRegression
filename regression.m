filename = 'data.txt';
M = csvread(filename);

hp  = M(:,1)  % Horespower
acc = M(:,2) % Acceleration

mdl = fitlm(hp, acc) % Linear Regression

scatter(acc,hp)
plot(mdl) % Plot the linear