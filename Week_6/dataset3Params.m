function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
sug_values = [0.01, 0,03, 0.1, 0.3, 1, 3, 10, 30];
min_err = +inf;
pred_err = 0;
for i=1:length(sug_values)
    for j=1:length(sug_values)
        model =  svmTrain(X, y, sug_values(i), @(x1, x2) gaussianKernel(x1, x2, sug_values(j))); 
        predictions = svmPredict(model, Xval);
        pred_err=mean(double(predictions ~= yval));
        if (pred_err < min_err) 
            C = sug_values(i);
            sigma = sug_values(j);
            min_err = pred_err;
        end
    end
end



% =========================================================================

end
