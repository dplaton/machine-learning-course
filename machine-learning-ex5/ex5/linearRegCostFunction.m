function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% unregulated part:
% first we compute the cost function (we did this in ex1 already)
h = X * theta;
J_unreg = (1/(2*m)) * sum((h-y).^2);
% ...then the gradient (still in ex1)
grad_unreg = (1/m) * (X' * (h - y));

% now, let's regulate things
%ignore theta(1) from reg
theta(1) = 0;
reg_term = (lambda / (2*m)) * (theta' * theta);
J = J_unreg + reg_term;
grad =  grad_unreg + ((lambda/m) * theta);

% =========================================================================

grad = grad(:);

end
