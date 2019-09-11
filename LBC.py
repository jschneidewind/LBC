from numpy import exp, atleast_1d, searchsorted, linspace, r_, c_, random, ones, log, abs
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def logistic(c, m, spn, x):
	return (c/(1 + exp(spn*(x - m))))

def residual(p, m, spn, x, y):
	y_poly = poly(p[:-1], x)
	y_log = logistic(p[-1], m, spn, x)
	res = y + y_log - y_poly
	return res

def residual_weighing(p, m, spn, x, y, idx_s, pre_weight_factor, post_weight_factor):
	y_poly = poly(p[:-1], x)
	y_log = logistic(p[-1], m, spn, x)
	weight = ones(len(x))
	weight[:idx_s] = pre_weight_factor
	weight[idx_s:] = post_weight_factor
	res = (y + y_log - y_poly) * weight
	return res

def find_nearest(array, values): # Moved vectorized searchsorted outside of loop, tidied up conditionals, removed math dependency
    
    if array.ndim != 1:
        array_1d = array[:,0]
    else:
        array_1d = array

    values = atleast_1d(values)
    values_idx = searchsorted(array_1d, values, side= "left")
    hits = []

    for i, idx in enumerate(values_idx):
        if idx == len(array_1d):
            hits.append(idx-1)
        elif abs(values[i] - array_1d[idx-1]) < abs(values[i] - array_1d[idx]):
            hits.append(idx-1)
        else:
            hits.append(idx)
    return(hits)

def first_order(c0, k, t):
	return 1 - (c0 * exp(-k*t))

def poly(a, x):
	y = a[0] * x**0
	for i in range(1, len(a)):
		y += a[i] * x**i
	return y

def synthetic_data(lower_l, upper_l, no_dp, c0, k, a, sigma):
	''' Generates synthetic data with polynomial (order determined by length of parameter vector a), over range defiend by lower_l and upper_l, 
	with variable number of datapoints (no_dp), variable signal characteristics (c0, k) and different levels of noise (sigma),
	returns 2d array with synthetic data
	Use e.g. lower_l = -2, upper_l = 4, no_dp = 100, c0 = 1., k = 4., a = [0.04, 0.2, -0.07], sigma = 0.02'''

	x_complete = linspace(lower_l, upper_l, no_dp)
	idx = find_nearest(x_complete, 0)[0]

	x_baseline = x_complete[:idx]
	x_signal = x_complete[idx:]

	signal = first_order(c0, k, x_signal) + poly(a, x_signal)
	baseline = poly(a, x_baseline)	
	noise = random.normal(0, sigma, no_dp)
	
	x = r_[x_baseline, x_signal]	
	y = r_[baseline, signal] + noise

	return c_[x, y]

def elevator_function_fitting(data, start, end, order_poly, pre_weight_factor = 1., post_weight_factor = 1., plotting = False):
	'''Perform LBC by providing data (2d numpy array with x values in column 0 and y values in column 1), 
	signal start and end point (in x units, not index) and the order of the polynomial used for baseline fitting;
	returns 2d array with baseline corrected data '''

	idx = find_nearest(data, (start, end))
	x = r_[data[:,0][:idx[0]], data[:,0][idx[1]:]] # Defining pre and post signal intervals
	y = r_[data[:,1][:idx[0]], data[:,1][idx[1]:]]

	m = (data[:,0][idx[0]] + data[:,0][idx[1]]) / 2  # sigmoidal midpoint midway between pre- and post-signal intervals
	x_75 = (data[:,0][idx[1]] - m) / 2
	spn = log(1.0/999999.0) / x_75  # logistic growth rate, defined so that logistic function reaches 99.9999% of maximum value at 75th percentile point between pre and post signal intervals
	p_guess = ones(order_poly+2)*0.1

	if pre_weight_factor != 1. or post_weight_factor != 1.:
		p = least_squares(fun=residual_weighing, x0=p_guess, args=(m, spn, x, y, idx[0], pre_weight_factor, post_weight_factor))
	else:
		p = least_squares(fun=residual, x0=p_guess, args=(m, spn, x, y))
	p_solved = p.x

	y_baseline = poly(p_solved[:-1], data[:,0]) 
	y_fit = y_baseline - logistic(p_solved[-1], m, spn, data[:,0])
	y_corrected = data[:,1] - y_baseline

	if plotting == True:
		plt.plot(x, y, '.', markersize = 2.0, color = 'blue')
		plt.plot(data[:,0], y_fit, linewidth = 1.0, color = 'red')
		plt.plot(data[:,0], y_baseline, linewidth = 1.0, color = 'green')
		plt.plot(data[:,0], data[:,1], linewidth = 0.2, color = 'blue')
		plt.show()

	return c_[data[:,0], y_corrected]

def main():
	data = synthetic_data(lower_l = -2., upper_l = 4., no_dp = 1000, c0 = 1., k = 4., a = [0.4, 0.05, -0.007], sigma = 0.02)
	find_nearest(data, (-2., 1.0001, 5.)) # should hit all conditionals of find_nearest
	data_corrected = elevator_function_fitting(data, start = 0., end = 2., order_poly = 2, plotting = True)
	data_corrected_weighted = elevator_function_fitting(data, start = 0., end = 2., order_poly = 2, pre_weight_factor = 2., plotting = False)

if __name__ == "__main__":
	main()



