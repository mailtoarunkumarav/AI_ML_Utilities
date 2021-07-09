from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import datetime
np.random.seed(200)


class SimpleGaussianProcess:

    def __init__(self, kernel_type, lengthscale, signal_variance, obj_function):
        self.kernel_type = kernel_type
        self.lengthscale = lengthscale
        self.signal_variance = signal_variance
        self.obj_function = obj_function

    def true_function(self, x):

        if self.obj_function == "Benchmark":
            value = np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + 1 / (x ** 2 + 1)
        elif self.obj_function == "Sin":
            value = np.sin(x)
        return value.flatten()

    # Define the kernel
    def kernel(self, x1, x2):
        # Squared exponential kernel
        if self.kernel_type == "SQ_EXP":
            value = self.sq_exp_kernel(x1, x2)
        # Matern Kernel
        elif self.kernel_type == "MAT":
            value = self.matern3_kernel(x1, x2)
        # Custom implementation of kernel
        elif self.kernel_type == "custom":
            value = self.custom_kernel(x1, x2)
        return value

    def sq_exp_kernel(self,x1, x2):
        sqdist = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
        value = (self.signal_variance ** 2) * np.exp(-.5 * (1 / self.lengthscale) * sqdist)
        return value

    def matern3_kernel(self, x1, x2):
        kernel_mat = np.zeros(shape=(len(x1), len(x2)))
        for i in np.arange(len(x1)):
            for j in np.arange(len(x2)):
                difference = (x1[i, :] - x2[j, :])
                l2_difference = np.sqrt(np.dot(difference, difference.T))
                each_kernel_val = (self.signal_variance ** 2) * (1 + (np.sqrt(3)*l2_difference/self.lengthscale)) * \
                                  (np.exp((-1 * np.sqrt(3) / self.lengthscale) * l2_difference))
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat

    def custom_kernel(self, x1, x2):
        return np.dot(x1, x2.T)+0.005

    def run_gaussian(self, number_observations_X, number_test_points_Xs, xmin, xmax, ymin, ymax, plot_samples):

        # Add noise to the model
        noise = 0.001

        # Number of Prior and POosterior samples for plotting GP prior and Posterior
        number_prior_function_samples = number_posterior_function_samples = 5

        # Generate random observations
        X = np.random.uniform(xmin, xmax, size=(number_observations_X, 1))
        y = self.true_function(X) + noise * np.random.randn(number_observations_X)

        # Compute GP Mean, Standard Deviation and Variance
        K_x_x = self.kernel(X, X)
        L_x_x = np.linalg.cholesky(K_x_x + noise * np.eye(number_observations_X))

        Xs = np.linspace(xmin, xmax, number_test_points_Xs).reshape(-1, 1)
        L_k_x_xs = np.linalg.solve(L_x_x, self.kernel(X, Xs))

        # Mean Calculation
        mean = np.dot(L_k_x_xs.T, np.linalg.solve(L_x_x, y))

        # Variance and Std Dev Calculation
        K_Xs_Xs = self.kernel(Xs, Xs)
        variance = np.diag(K_Xs_Xs) - np.sum(L_k_x_xs ** 2, axis=0)
        std_dev = np.sqrt(variance)

        plt.figure("GP Mean and Variance")
        plt.title('GP Mean and +- 2 standard deviations ')
        plt.clf()
        plt.plot(X, y, 'ro', ms=5)
        plt.plot(Xs, self.true_function(Xs), 'b-')
        plt.gca().fill_between(Xs.flat, mean - 2 * std_dev, mean + 2 * std_dev, color="#726E6D", alpha=0.2)
        plt.plot(Xs, mean, 'g--', lw=2)
        plt.savefig('gp_posterior_distribution.png', bbox_inches='tight')

        plt.axis([xmin, xmax, ymin, ymax])

        if plot_samples:
            # Draw GP Prior Samples
            L_x_x = np.linalg.cholesky(K_Xs_Xs + 1e-6 * np.eye(number_test_points_Xs))
            f_prior = np.dot(L_x_x, np.random.normal(size=(number_test_points_Xs, number_prior_function_samples)))
            plt.figure("GP Prior Function Samples")
            plt.title('GP Prior  Function Samples')
            plt.clf()
            plt.plot(Xs, f_prior)
            plt.axis([xmin, xmax, ymin, ymax])
            plt.savefig('gp_prior_samples.pdf', bbox_inches='tight')

            # Draw GP Posterior samples
            L_x_x = np.linalg.cholesky(K_Xs_Xs + 1e-6 * np.eye(number_test_points_Xs) - np.dot(L_k_x_xs.T, L_k_x_xs))
            f_posterior = mean.reshape(-1, 1) + np.dot(L_x_x,
                                                       np.random.normal(size=(number_test_points_Xs, number_posterior_function_samples)))
            plt.figure("Posterior Function Samples")
            plt.title('GP Posterior Function Samples')
            plt.clf()
            plt.plot(Xs, f_posterior)
            plt.axis([xmin, xmax, ymin, ymax])
            plt.savefig('gp_posterior_samples.pdf', bbox_inches='tight')

def gp_wrapper():

    # Number of Training points
    number_observations_X = 10

    # Number of Test points.
    number_test_points_Xs = 100

    # kernel_type is the kernel to be used. Modify
    # Implemented values: (1) kernel_type = "SQ_EXP" - for Squared Exponential kernel, (2) kernel_type = "custom" - for custom kernels
    # kernel_type = "custom" # Replace the current linear kernel implementation in custom_kernel() with your own implementation
    # kernel_type = "MAT"
    kernel_type = "SQ_EXP"

    # Characteristic Lengthscale
    lengthscale = 0.7
    # Signal Variance
    signal_variance = 1

    # Objective Function Type - objective function being modelled
    # (1) obj_function = "Benchmark" - for modelling benchmark objective function, (2) obj_function = "Sin" - for modelling sin function
    obj_function = "Benchmark"
    # obj_function = "Sin"

    # Define input space boundaries
    xmin = 0
    xmax = 10
    ymin = -2
    ymax = 3

    # Do you want GP prior and GP Posterior samples
    plot_samples = True

    gp_obj = SimpleGaussianProcess(kernel_type, lengthscale, signal_variance, obj_function)
    gp_obj.run_gaussian(number_observations_X, number_test_points_Xs, xmin, xmax, ymin, ymax, plot_samples)
    plt.show()


if __name__ == "__main__":
    timenow = datetime.datetime.now()
    stamp = timenow.strftime("%H%M%S_%d%m%Y")
    print("Start time: ", stamp)
    gp_wrapper()

