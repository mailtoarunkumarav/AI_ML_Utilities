from scipy.stats import norm
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Class to handle the Acquisition Functions related tasks required for the Bayesian Optimization
class AcquisitionFunction():

    # Initializing the parameters required for the ACQ functions
    def __init__(self, acq_type, number_of_restarts, number_of_dimensions, kappa, epsilon1, epsilon2):
        self.acq_type = acq_type
        self.number_of_restarts = number_of_restarts
        self.number_of_dimensions = number_of_dimensions
        self.kappa = kappa
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2

    # Method to set the type of ACQ function to be used for the Optimization process
    def set_acq_func_type(self, type):
        self.acq_type = type

    # Expected improvement ACQ function
    def expected_improvement(self, mean, std_dev, y_max):
        with np.errstate(divide='ignore'):
            z_value = (mean - y_max - self.epsilon2) / std_dev
            zpdf = norm.pdf(z_value)
            zcdf = norm.cdf(z_value)
            ei_acq_func = np.dot(zcdf, (mean - y_max - self.epsilon2)) + np.dot(std_dev, zpdf)
            # ei_acq_func[std_dev==0] == 0
        return  ei_acq_func

    # Probability improvement ACQ function
    def probability_improvement(self, mean, std_dev, y_max):
        z_value = (mean - y_max - self.epsilon1) / std_dev
        zcdf = norm.cdf(z_value)
        return zcdf

    # UCB ACQ function
    def ucb_acq_func(self, mean, std_dev, iteration_count):

        with np.errstate(divide='ignore') or np.errstate(invalid='ignore'):

            # Constant parameters to be used while maximizing the ACQ function
            delta = 0.1; b = 1;a = 1;r = 1;v=1;
            # d - number of dimensions
            d = self.number_of_dimensions

            # Different values for beta
            # formula beta1 = 2log(t22π2/ (3δ)) + 2dlogt2dbr * sqrt(log(4da / δ))
            # beta1 = 2 * np.log((iteration_count ** 2) * (2 * (np.pi ** 2)) * (1 / (3 * delta))) +\
            #         (2 * d) * np.log((iteration_count ** 2) * d * b * r * (np.sqrt(np.log(4 * d * a * (1 / delta)))))

            # # formula beta2 = 2log(dt2π2/6δ)
            # beta2 = 2 * np.log(d*(iteration_count**2)*(np.pi**2)*(1/(6*delta)))

            # formula beta3 = 2 log(t((d/2)+2) (π^2/3δ))
            beta3 = 2*np.log((iteration_count**((d/2)+2))*(np.pi**2)*(1/(3*delta)))
            self.kappa = np.sqrt(v * beta3)
            # self.kappa = 0.1 * beta3
            result = mean + self.kappa * std_dev
            return result

    # Helper method to invoke the EI acquisition function
    def expected_improvement_util(self, x, y_max, gp_obj):


        with np.errstate(divide='ignore') or np.errstate(invalid='ignore'):

            # Use Gaussian Process to predict the values for mean and variance at given x
            mean, variance, fprior, f_post = gp_obj.gaussian_predict(np.matrix(x))
            std_dev = np.sqrt(variance)
            result = self.expected_improvement(mean, std_dev, y_max)
            # Since scipy.minimize function is used to find the minima and so converting it to maxima by * with -1
            return -1 * result

    def probability_improvement_util(self, x, y_max, gp_obj):
        with np.errstate(divide='ignore') or np.errstate(invalid='ignore'):
            # Use Gaussian Process to predict the values for mean and variance at given x
            mean, variance, fprior, f_post = gp_obj.gaussian_predict(np.matrix(x))
            # mean, variance, fprior, f_post = gp_obj.gaussian_predict(np.array(x))
            std_dev = np.sqrt(variance)
            result = self.probability_improvement(mean, std_dev, y_max)
            # Since scipy.minimize function is used to find the minima and so converting it to maxima by * with -1
            return -1 * result

    def upper_confidence_bound_util(self, x, gp_obj, iteration_count ):
        with np.errstate(divide='ignore') or np.errstate(invalid='ignore'):
            # Use Gaussian Process to predict the values for mean and variance at given x
            mean, variance, fprior, f_post = gp_obj.gaussian_predict(np.matrix(x))
            std_dev = np.sqrt(variance)
            result = self.ucb_acq_func(mean, std_dev, iteration_count)
            # Since scipy.minimize function is used to find the minima and so converting it to maxima by * with -1
            return -1 * result


    # Method to maximize the ACQ function as specified the user
    def max_acq_func(self, gp_obj, Xs, ys, iteration_count):

        # Initialize the xmax value and the function values to zeroes
        x_max_value = np.zeros(gp_obj.number_of_dimensions)
        fmax = 0

        # Temporary data structures to store xmax's and function values using scipy.minimize
        tempmax_x=[]
        tempfvals=[]

        # Data structure to create the initialization points for the scipy.minimize minimiser
        random_points = []
        starting_points = []

        # Depending on the number of dimensions and bounds, generate random multiple initialization points
        for dim in np.arange(gp_obj.number_of_dimensions):
            random_data_point_each_dim = np.random.uniform(gp_obj.bounds[dim][0], gp_obj.bounds[dim][1],
                                                           self.number_of_restarts).reshape(1,
                                                                                        self.number_of_restarts)
            random_points.append(random_data_point_each_dim)

        # Vertically stack the arrays of randomly generated starting points as a matrix
        random_points = np.vstack(random_points)

        # Reformat the generated random starting points in the form [x1 x2].T for the specified number of restarts
        for sample_num in np.arange(gp_obj.number_of_observed_samples):
            array = []
            for dim_count in np.arange(gp_obj.number_of_dimensions):
                array.append(random_points[dim_count, sample_num])
                starting_points.append(array)
        starting_points = np.vstack(starting_points)

        # Find maxima of the ACQ function using PI
        if self.acq_type == 'pi':

            # Obtain the maximum value of the unknown function from the samples observed already
            y_max = gp_obj.y.max()
            print("ACQ Function : PI ")

            # Obtain the maxima of ACQ function by starting the optimization at different start points
            for starting_point in starting_points:

                # Find the maxima in the bounds specified for the PI ACQ function
                max_x = opt.minimize(lambda x: self.probability_improvement_util(x, y_max, gp_obj), starting_point,
                                     method='L-BFGS-B',
                                     bounds=gp_obj.bounds)

                # Use gaussian process to predict the mean and variances for the maximum point identified
                mean, variance, f_prior, f_post = gp_obj.gaussian_predict(np.matrix(max_x['x']))
                std_dev = np.sqrt(variance)
                fvalue = self.probability_improvement(mean, std_dev, y_max)

                # Store the maxima of ACQ function and the corresponding maxima (required for debugging)
                tempmax_x.append(max_x['x'])
                tempfvals.append(fvalue)

                # Compare the values obtained in the current run to find the best value overall and store accordingly
                if (fvalue > fmax):
                    print("New best Fval: ",fvalue," found at: ", max_x['x'])
                    x_max_value = max_x['x']
                    fmax = fvalue

            print("PI Best is ", fmax, "at ", x_max_value)

            # Calculate the ACQ function values at each of the unseen data points to plot the ACQ function
            with np.errstate(invalid='ignore'):
                mean, variance, f_prior, f_post = gp_obj.gaussian_predict(Xs)
                std_dev = np.sqrt(variance)
                acq_func_values = self.probability_improvement(mean, std_dev, y_max)

            # used to verify if the maxima value is really found at 0
            # if(x_max_value==[0]):
            #     print('\n\ntemp fvalues and xmax',tempfvals, tempmax_x)

        # Find maxima of the ACQ function using UCB
        elif self.acq_type == "ucb":
            print ("ACQ Function : UCB ")

            # Obtain the maxima of ACQ function by starting the optimization at different start points
            for starting_point in starting_points:

                # Find the maxima in the bounds specified for the UCB ACQ function
                max_x = opt.minimize(lambda x: self.upper_confidence_bound_util(x, gp_obj, iteration_count),starting_point ,
                                     method='L-BFGS-B',
                                     bounds=gp_obj.bounds)

                # Use gaussian process to predict the mean and variances for the maximum point identified
                mean, variance, f_prior, f_post = gp_obj.gaussian_predict(np.matrix(max_x['x']))
                std_dev = np.sqrt(variance)
                fvalue = self.ucb_acq_func(mean, std_dev, iteration_count)

                # Store the maxima of ACQ function and the corresponding maxima (required for debugging)
                tempmax_x.append(max_x['x'])
                tempfvals.append(fvalue)

                # Compare the values obtained in the current run to find the best value overall and store accordingly
                if fvalue > fmax:
                    print("New best Fval: ", fvalue, " found at: ", max_x['x'])
                    x_max_value = max_x['x']
                    fmax = fvalue

            print("UCB Best is ", fmax, "at ", x_max_value)

            # Calculate the ACQ function values at each of the unseen data points to plot the ACQ function
            with np.errstate(invalid='ignore'):
                mean, variance, f_prior, f_post = gp_obj.gaussian_predict(Xs)
                std_dev = np.sqrt(variance)
                acq_func_values = self.ucb_acq_func(mean, std_dev, iteration_count)

        # Find maxima of the ACQ function using EI
        elif self.acq_type == 'ei':

            # Obtain the maximum value of the unknown function from the samples observed already
            y_max = gp_obj.y.max()
            print("ACQ Function : EI ")

            # Obtain the maxima of the ACQ function by starting the optimization at different start points
            for starting_point in starting_points:

                # Find the maxima in the bounds specified for the PI ACQ function
                max_x = opt.minimize(lambda x: self.expected_improvement_util(x, y_max, gp_obj), starting_point,
                                     method='L-BFGS-B',
                                     bounds=gp_obj.bounds)

                # Use gaussian process to predict mean and variances for the maximum point identified
                mean, variance, f_prior, f_post = gp_obj.gaussian_predict(np.matrix(max_x['x']))
                std_dev = np.sqrt(variance)
                fvalue = self.expected_improvement(mean, std_dev, y_max)

                # Store the maxima of ACQ function and the corresponding maxima (required for debugging)
                tempmax_x.append(max_x['x'])
                tempfvals.append(fvalue)

                # Compare the values obtained in the current run to find the best value overall and store accordingly
                if fvalue > fmax:
                    print("New best Fval: ", fvalue, " found at: ", max_x['x'])
                    x_max_value = max_x['x']
                    fmax = fvalue

            print("EI Best is ", fmax, "at ", x_max_value)

            # Calculate the ACQ function values at each of the unseen data points to plot the ACQ function
            with np.errstate(invalid='ignore'):
                mean, variance, f_prior, f_post = gp_obj.gaussian_predict(Xs)
                std_dev = np.sqrt(variance)
                acq_func_values = self.expected_improvement(mean, std_dev, y_max)

            # used to verify if the maxima value is really found at 0
            # if(x_max_value==[0]):
            #     print(tempfvals, tempmax_x)

        return x_max_value, acq_func_values

    # Helper method to plot the values found for the specified ACQ function at unseen data points
    def plot_acquisition_function(self, count, Xs, acq_func_values, plot_axis):

        # Set the parameters of the ACQ functions plot
        plt.figure('Acquisition Function - ' + str(count))
        plt.clf()
        plt.plot(Xs, acq_func_values)
        plt.axis(plot_axis)
        plt.title('Acquisition Function')
        plt.savefig('acq', bbox_inches='tight')

