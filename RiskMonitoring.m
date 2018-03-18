%% NSE Hackathon : RiskMonitoring System Using Support Vector Machines
%  Author: Mangesh More
%  
%  Instructions
%  ------------
%  This file contains code that helps you get started on RiskMonitoring System using Vector Machines
%  You will need to complete the following functions in this exericse:
%     .m


%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the MWPL and Rollover data and the third column contains the label.
data = load('RiskMonitoring.txt');
X = data(:, [1, 2]); y = data(:, 3);

%% =============== Part 1: Loading and Visualizing Data ================

fprintf(['Part 1: Plotting data with + indicating ''Flagged'' examples and o indicating ''AllClear''.\n']);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

plotData(X, y);

% Display Labels and Legend 
hold on;
xlabel('MWPL')
ylabel('Rollover')
legend('Flagged', 'AllClear')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ==================== Part 2: Training Linear SVM ====================
%  The following code will train a linear SVM on the dataset and plot the decision boundary learned.

fprintf('\nPart 2: Training Linear SVM ...\n')
 
C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);
% Display Labels and Legend 
hold on;
xlabel('MWPL')
ylabel('Rollover')
legend('Flagged', 'AllClear')
hold off
fprintf('Changing C value above and affects how the decision boundary varies (e.g., C = 1000).\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Part 3: Implementing Gaussian Kernel on Sample non Linear Data===============
%  Implementing the Gaussian kernel to use with the SVM.
%
fprintf('\nPart 3: Evaluating the Gaussian Kernel on Sample non Linear Data...\n')

x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);

fprintf(['\n Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f :' ...
         '\n\t%f\n(for sigma = 2, this value should be about 0.324652)\n'], sigma, sim);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Part 4: Visualizing Sample NonLinear Dataset  ================
%  The following code will load the Sample non linear dataset into your environment and plot the data. 

fprintf('Part 4: Loading and Visualizing Sample NonLinear Dataset ...\n')

% Load from Sample NonLinear Dataset: 
load('NonLinear.mat');

% Plot training data
plotData(X, y);

% Display Labels and Legend 
hold on;
xlabel('xSample')
ylabel('ySample')
legend('Flagged', 'AllClear')
hold off

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========== Part 5: Training SVM with Gaussian Kernel on Sample NonLinear Dataset ==========
%  After implemention the kernel, use it to train the SVM classifier.
% 
fprintf('\nPart 5: Training SVM with Gaussian Kernel on Sample NonLinear Dataset (this may take 1 to 2 minutes) ...\n');

% Load from NonLinear: 
% You will have X, y in your environment
load('NonLinear.mat');

% SVM Parameters
C = 1; sigma = 0.1;

% We set the tolerance and max_passes lower here so that the code will run
% faster. However, in practice, you will want to run the training to
% convergence.
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(X, y, model);

% Display Labels and Legend 
hold on;
xlabel('xSample')
ylabel('ySample')
legend('Flagged', 'AllClear')
hold off

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Part 6: Visualizing Sample NonLinear2 Dataset  ================
%  The following code will load the NonLinear2 and plot the data. 

fprintf('Part 6: Loading and Visualizing Sample NonLinear2 Dataset ...\n')

% Load from NonLinear2: 
% You will have X, y in your environment
load('NonLinear2.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========== Part 7: Training SVM with Gaussian Kernel ==========
% SVM Parameters
C = 1; sigma = 0.1;

[C, sigma] = dataset3Params(X, y, Xval, yval);

% Train the SVM
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);

% Display Labels and Legend 
hold on;
xlabel('MWPL')
ylabel('Rollover')
legend('Flagged', 'AllClear')
hold off

fprintf('Program paused. Press enter to continue.\n');
pause;