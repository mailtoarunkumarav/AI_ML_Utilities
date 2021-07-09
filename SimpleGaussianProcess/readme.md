1. This module implements a Gaussian Process (GP) using the observations supplied (see point 8. for important configurable parameters).

2. The configurable parameters are provided in "gp_wrapper()" method of SimpleGaussianProcess.py

3. Users can configure the kernel function to be used, along with its hyperparameters.  Currently, it supports Squared Exponential (SE) kernel and Matern kernel with \nu=3/2. See "kernel_type" parameter for more information. 

4. Users can construct their own kernel by providing its mathematical formulation in custom_kernel(x1, x2) method.

5. The current implementation doesn't support the log-likelihood based tuning of the kernel hyperparameters (pre-selected lengthscale and signal variance parameters).

6. Currently, the following objective functions are considered for modelling (1) 1D Benchmark function (2) Sin function. Users can define their choice of objective function to model in "true_function(x)" method of Simple GaussianProcess.py. See "obj_function" parameter for more information.

7. Plots: (a). GP posterior distribution, (b). GP prior samples, (c) GP Posterior samples. GP prior and posterior samples can be skipped from plotting by setting plot_samples = False in gp_wrapper()

8. List of configurable parameters in gp_wrapper() method: (a). Number of observations (X),  (b) Number of test points (Xs), (c). Kernel type, (d). Objective function being modelled, (e) Characteristic length scale, (f). Signal variance, (g). Input space boundaries (used for plotting and generating random observations).
