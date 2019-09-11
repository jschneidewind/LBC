import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

def find_nearest(array, values): 

    if array.ndim != 1:
        array_1d = array[:,0]
    else:
        array_1d = array

    values = np.atleast_1d(values)
    values_idx = np.searchsorted(array_1d, values, side= "left")
    hits = []

    for i, idx in enumerate(values_idx):
        if idx == len(array_1d):
            hits.append(idx-1)
        elif np.abs(values[i] - array_1d[idx-1]) < np.abs(values[i] - array_1d[idx]):
            hits.append(idx-1)
        else:
            hits.append(idx)
    return(hits)

def first_order(c0, k, t):

	return 1 - (c0 * np.exp(-k*t))

def poly(a, x):

	y = a[0] * x**0

	for i in range(1, len(a)):
		y += a[i] * x**i

	return y

def synthetic_data(lower_l, upper_l, no_dp, c0, k, a, sigma):

	x_complete = np.linspace(lower_l, upper_l, no_dp)
	idx = find_nearest(x_complete, 0)[0]

	x_baseline = x_complete[:idx]
	x_signal = x_complete[idx:]

	signal = first_order(c0, k, x_signal) + poly(a, x_signal)
	baseline = poly(a, x_baseline)
	baseline_complete = poly(a, x_complete)

	pure_signal = first_order(c0, k, x_signal) + np.random.normal(0, sigma, len(x_signal))
	
	noise = np.random.normal(0, sigma, no_dp)
	
	x = np.r_[x_baseline, x_signal]	
	y = np.r_[baseline, signal] + noise

	return np.c_[x, y], np.c_[x_complete, baseline_complete], np.c_[x_signal, pure_signal]

def logistic(c, m, spn, x):

	return (c/(1 + np.exp(spn*(x - m))))

def residual(p, m, spn, x, y):

	y_poly = poly(p[:-1], x)
	y_log = logistic(p[-1], m, spn, x)
	res = y + y_log - y_poly

	return res

def residual_pre_signal(p, x, y):

	y_poly = poly(p, x)
	res = y - y_poly

	return res

def residual_first_order(p, x, y):

	y_fit = first_order(1., p[0], x)
	res = y - y_fit

	return res

def elevator_function_fitting(data, start, end, order_poly):
	'''LBC function'''

	idx = find_nearest(data, (start, end))
	x = np.r_[data[:,0][:idx[0]], data[:,0][idx[1]:]]
	y = np.r_[data[:,1][:idx[0]], data[:,1][idx[1]:]]

	m = (data[:,0][idx[0]] + data[:,0][idx[1]]) / 2
	x_mid = (data[:,0][idx[1]] - m) / 2
	spn = np.log(1.0/999999.0) / x_mid

	p_guess = np.ones(order_poly+2)*0.01

	p = least_squares(fun=residual, x0=p_guess, args=(m, spn, x, y))
	p_solved = p.x

	y_baseline = poly(p_solved[:-1], data[:,0]) 
	y_fit = y_baseline - logistic(p_solved[-1], m, spn, data[:,0])
	y_corrected = data[:,1] - y_baseline

	return np.c_[data[:,0], y_corrected], p_solved[-1], p_solved[:-1]

def pre_signal_fitting(data, end, order_poly):

	idx = find_nearest(data, end)
	x = data[:,0][:idx[0]]
	y = data[:,1][:idx[0]]

	p_guess = np.ones(order_poly+1)
	p = least_squares(fun=residual_pre_signal, x0=p_guess, args=(x, y))
	p_solved = p.x

	y_baseline = poly(p_solved, data[:,0])
	y_corrected = data[:,1] - y_baseline

	return np.c_[data[:,0], y_corrected]

def first_order_fitting(data):

	idx = find_nearest(data, 0)[0]

	x = data[:,0][idx:]
	y = data[:,1][idx:]

	p_guess = np.ones(1)
	p = least_squares(fun=residual_first_order, x0=p_guess, args=(x, y))
	p_solved = np.array([1., p.x])

	y_fit = first_order(p_solved[0], p_solved[1], x)

	return p.x

def variable_first_order_fitting(samples, no_dp, sigma, lower_l = -2., upper_l = 4., c0 = 1., k = 4., a = [0.4, 0.05, -0.007], start = -0.005, end = 2., order_poly = 2):
	'''Performs fitting of first order, integrated rate law to baseline corrected data obtained using different baseline
	correction methods with variable total number of datapoints and variable sigma of Gaussian noise'''

	k_elevator = []
	k_pre_signal = []
	k_pure_signal = []

	for i in range(samples):

		data, baseline, pure_signal = synthetic_data(lower_l, upper_l, no_dp, c0, k, a, sigma)

		elevator_data, p, c = elevator_function_fitting(data, start, end, order_poly)
		pre_signal_data = pre_signal_fitting(data, start, order_poly)

		p_elevator = first_order_fitting(elevator_data)
		p_pre_signal = first_order_fitting(pre_signal_data)
		p_pure_signal = first_order_fitting(pure_signal)

		k_elevator.append(p_elevator)
		k_pre_signal.append(p_pre_signal)
		k_pure_signal.append(p_pure_signal)

	arr_elevator = np.asarray(k_elevator)
	arr_pre_signal = np.asarray(k_pre_signal)
	arr_pure_signal = np.asarray(k_pure_signal)

	return arr_elevator, arr_pre_signal, arr_pure_signal

def get_failed_fittings(data, threshold):

	fails = np.where(data < threshold)[0]
	return len(fails)

def variable_performance_test(samples, order_poly, noise_level, no_dp, plotting = True):
	'''Function evaluates performance of LBC for recovery of O_s and baseline parameters for different baseline shapes, 
	nosie levels (sigma of Gaussian noise, and total number of datapoints. Function used to generate SI Figure 2.'''


	if len(order_poly) > 1:
		variable = order_poly
		pos = 0

	if len(noise_level) > 1:
		variable = noise_level
		pos = 1

	if len(no_dp) > 1:
		variable = no_dp
		pos = 2

	c_list = []
	p_list = []
	settings = np.array([0, 0, 0])

	for i in range(len(variable)):
	
		settings[pos] = i
		op = order_poly[settings[0]]
		nl = noise_level[settings[1]]
		nd = no_dp[settings[2]]

		for _ in range(samples):

			a = np.random.uniform(-0.5, 0.5, op+1)
			data, baseline, pure_signal = synthetic_data(lower_l = -2., upper_l = 4., no_dp = nd, c0 = 1., k = 4., a = a, sigma = nl)
		
			elevator_data, c, p = elevator_function_fitting(data, -0.005, 2.0, op)

			c_list.append(-c)
			p_list.append(np.sqrt((1./len(a))*np.sum((a - p)**2)))

	arr_c = np.asarray(c_list)
	arr_p = np.asarray(p_list)

	c_np = np.reshape(arr_c, (-1, samples))
	p_np = np.reshape(arr_p, (-1, samples))

	result = np.stack((c_np, p_np))

	if plotting == True:
		plt.figure()
		plt.boxplot(c_np.T, whis = [5, 95], showfliers = False, patch_artist = True, labels = variable)
		plt.figure()
		plt.boxplot(result[1].T, whis = [5, 95], showfliers = False, patch_artist = True, labels = variable)

	return result

def first_order_fitting_performance(samples, lower_l = -2., upper_l = 4., no_dp = 1000, c0 = 1., k = 4., a = [0.4, 0.05, -0.007], sigma = 0.02, start = -0.005, end = 2., order_poly = 2, plotting = True):
	'''Function evaluates accuracy of k obtained by fitting first order, integrated rate law to baseline corrected data obtained
	using different baseline correction methods. Function was used to generate Figure 2 (manuscript)'''

	k_elevator = []
	k_pre_signal = []
	k_pure_signal = []

	for i in range(samples):

		data, baseline, pure_signal = synthetic_data(lower_l, upper_l, no_dp, c0, k, a, sigma)

		elevator_data, c, p = elevator_function_fitting(data, start, end, order_poly)
		pre_signal_data = pre_signal_fitting(data, start, order_poly)

		p_elevator = first_order_fitting(elevator_data)
		p_pre_signal = first_order_fitting(pre_signal_data)
		p_pure_signal = first_order_fitting(pure_signal)

		k_elevator.append(p_elevator)
		k_pre_signal.append(p_pre_signal)
		k_pure_signal.append(p_pure_signal)

	arr_elevator = np.asarray(k_elevator)
	arr_pre_signal = np.asarray(k_pre_signal)
	arr_pure_signal = np.asarray(k_pure_signal)

	result = np.c_[arr_elevator, arr_pre_signal, arr_pure_signal]

	if plotting == True:
		plt.figure()
		plt.hist(arr_pure_signal, alpha = 0.7, bins = np.arange(k-1., k+1., 0.02), histtype =u'step', fill = 'green', linewidth = 1., edgecolor = 'black')
		plt.hist(arr_pre_signal, alpha = 0.7, bins = np.arange(k-1., k+1., 0.02), histtype =u'step', fill = 'orange', linewidth = 1., edgecolor = 'black')
		plt.hist(arr_elevator, alpha = 0.7, bins = np.arange(k-1., k+1., 0.02), histtype =u'step', fill = 'blue', linewidth = 1., edgecolor = 'black')

		print('mean elevator:', np.mean(arr_elevator))
		print('mean pre signal:', np.mean(arr_pre_signal))
		print('mean pure signal:', np.mean(arr_pure_signal))
		print('std elevator:', np.std(arr_elevator))
		print('std pre signal:', np.std(arr_pre_signal))
		print('std pure signal:', np.std(arr_pure_signal))

	return result

def fitting_failure_performance(samples, no_dp, sigma, threshold = 1., plotting = True):
	'''Function evaluates the number of failed fittings for different total number of datapoints and different
	noise levels. Failed fittings refer to instances where fitting first order, integrated rate law to baseline
	corrected data yields unreasonable k. Function was used to generate SI Figure 3.'''


	if len(no_dp) > 1:
		variable = no_dp
		pos = 0

	if len(sigma) > 1:
		variable = sigma
		pos = 1

	ele_list = []
	pre_list = []
	pure_list = []
	settings = np.array([0, 0])

	for i in range(len(variable)):

		settings[pos] = i
		nd = no_dp[settings[0]]
		nl = sigma[settings[1]]

		ele, pre, pure = variable_first_order_fitting(samples, nd, nl)

		ele_list.append(get_failed_fittings(ele, threshold))
		pre_list.append(get_failed_fittings(pre, threshold))
		pure_list.append(get_failed_fittings(pure, threshold))

	ele_np = 100. * np.asarray(ele_list) / samples 
	pre_np = 100. * np.asarray(pre_list) / samples
	pure_np = 100. * np.asarray(pure_list) / samples

	if plotting == True:
		fig, ax = plt.subplots()
		width = 0.35
		x = np.arange(len(variable))

		ax.bar(x - width/2, ele_np, width, label = 'LBC')
		ax.bar(x + width/2, pre_np, width, label = 'Pre-Signal Fitting')

		ax.set_ylabel('% Failed Fittings')
		ax.set_xticks(x)
		ax.set_xticklabels(variable)
		ax.set_xlabel('$ \sigma $ of Gaussian Noise')

		legend_a = ax.legend()

	return ele_np, pre_np, pure_np, variable

def main():

	variable_performance_test(samples = 10, order_poly = np.array([2]), noise_level = np.array([0.001, 0.01, 0.02, 0.03, 0.1]), no_dp = np.array([100]))
	first_order_fitting_performance(20)
	fitting_failure_performance(200, np.array([1000]), np.array([0.01, 0.05, 0.1, 0.2, 0.5]))
	plt.show()

if __name__ == '__main__':
	main()