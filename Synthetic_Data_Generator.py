import numpy as np
import find_nearest as fn

def first_order(c0, k, t):

	return 1 - (c0 * np.exp(-k*t))

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

	x_complete = np.linspace(lower_l, upper_l, no_dp)
	idx = fn.find_nearest(x_complete, 0)[0]

	x_baseline = x_complete[:idx]
	x_signal = x_complete[idx:]

	signal = first_order(c0, k, x_signal) + poly(a, x_signal)
	baseline = poly(a, x_baseline)	
	noise = np.random.normal(0, sigma, no_dp)
	
	x = np.r_[x_baseline, x_signal]	
	y = np.r_[baseline, signal] + noise

	return np.c_[x, y]
