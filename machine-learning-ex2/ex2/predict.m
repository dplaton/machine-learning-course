function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters.
%               You should set p to a vector of 0's and 1's
%
% Excerpt from a forum post on why don't we do theta'*X

% When you compute one hypothesis value, you use theta' * x, where x is a single training example as a vector.
% When you compute all the hypothesis values as a vector, you use X * theta. The dimensional analysis is (m x n) * (n x 1) = (m x 1), that is, a column vector containing one hypothesis value for each training example.

v = sigmoid(X*theta);
p = (v >= 0.5);
% =========================================================================


end
