function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
n = size(X, 2) + 1;
k = max(y);
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%covert y into binary vector using logical expression

y = y == [1:k];


%add 1s to X
a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);

%add 1s to a2
a2 = [ones(m, 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

%vectorized calculattion of cost
J = 1/m * (-y'*log(a3)-(1-y)'*log(1-a3));
J = trace(J);

%remove bias term
modTheta1 = Theta1;
modTheta1(:, 1)=0;
modTheta2 = Theta2;
modTheta2(:, 1)=0;

%calculate regularization term
regTerm = lambda/(2*m) * (trace(modTheta1*modTheta1') + trace(modTheta2*modTheta2'));

%cost and regularized term
J = J + regTerm;

%backpropagation
capDelta2 = 0;
capDelta1 = 0;


delta3 = a3 - y;
delta2 = delta3 * Theta2 .* (a2 .* (1-a2));
delta2 = delta2(:, 2:end);

capDelta2 = capDelta2 + delta3' * a2;
capDelta1 = capDelta1 + delta2' * a1;
%no regression on j-1 so replace first column with zeros
Theta2reg = Theta2;
Theta2reg(:, 1) = 0;
Theta1reg = Theta1;
Theta1reg(:, 1) = 0;


Theta2_grad = 1/m * capDelta2 + lambda/m * Theta2reg;
Theta1_grad = 1/m * capDelta1 + lambda/m * Theta1reg;





## non-vectorized solution
##for i = 1:m;
##  for j = 1:k;
##
##J = J + 1/m * (-y(i, j)*log(a3(i, j))-(1-y(i, j))*log(1-a3(i, j)));
##j = j + 1;
##
##end 
##i = i + 1;
##
##end 

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
