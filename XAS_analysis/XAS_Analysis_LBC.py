import matplotlib.pyplot as plt
import numpy as np
from larch import Interpreter, Group
from larch_plugins.xafs import pre_edge, mback
from scipy.optimize import least_squares
from scipy import interpolate
from numpy.linalg import inv

_larch = Interpreter(with_plugins = False)

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

def import_xdi(name):

	if name[-3:] == 'txt':
		data_raw = np.loadtxt(name)

		data = Group()
		data.energy = data_raw[:,2]
		data.mu = data_raw[:,8]/data_raw[:,5]

	else:

		data_raw = np.loadtxt(name)
	
		data = Group()
		data.energy = data_raw[:,0]
		data.mu = data_raw[:,1]
		data.i0 = data_raw[:,2]

	return data

def poly(a, x):

	y = a[0] * x**0

	for i in range(1, len(a)):
		y += a[i] * x**i

	return y

def expo(a, x):

	return a[0] * np.exp(a[1]*x)

def poly_expo(a, x):

	return expo(a[:2], x) + poly(a[2:], x)

def lorentzian(a, x):

	return a[0]/(1+((a[1]-x)/(a[2]/2))**2)

def gaussian(a, x):

	return a[0] * np.exp(-((x-a[1])**2)/(a[2]**2))

def voigt(a, x):

	return a[0] * lorentzian(a[1:], x) + (1 - a[0]) * gaussian(a[1:], x)

def voigt_mult(a, x):
	'''Sum of multiple Voigt functions with number of Voigt functions determined by length of a. Length of
	a has to be a multiple of 4.'''

	a_mat = np.reshape(a, (-1, 4))
	y = voigt(a_mat[0], x)

	for i in a_mat[1:,]:
		y = y + voigt(i, x)

	return y

def logistic(c, m, spn, x):

	return (c/(1 + np.exp(spn*(x - m))))

def residual(p, m, spn, x, y):

	y_poly = poly(p[:-1], x)
	y_log = logistic(p[-1], m, spn, x)
	res = y + y_log - y_poly

	return res

def residual_generic(p, x, y, function):
	'''Generic residual function using any function for fit.'''

	y_fit = function(p, x)
	res = y - y_fit

	return res

def larch_baseline(data, plot = False):
	'''Conventional baseline correction/normalization approach as implemented in Larch. Alternatively, MBACK
	algorithm implemented in Larch can be used'''

	pre_edge(data, _larch = _larch)
	print(data.edge_step)
	#mback(data, z=29, edge='K', order=4, _larch = _larch)

	if plot == True:
		plt.plot(data.energy, data.mu)
		plt.plot(data.energy, data.norm)
		plt.plot(data.energy, data.flat)

	return np.c_[data.energy, data.mu]

def elevator_function_fitting(data, start, end, order_poly, plot = False):
	'''LBC function returning normalized, baseline corrected spectra. Normalization is done using
	c parameter from logistic function.'''

	idx = find_nearest(data, (start, end))
	x = np.r_[data[:,0][:idx[0]], data[:,0][idx[1]:]]
	y = np.r_[data[:,1][:idx[0]], data[:,1][idx[1]:]]

	m = (data[:,0][idx[0]] + data[:,0][idx[1]]) / 2
	x_75 = (data[:,0][idx[1]] - m) / 2
	spn = np.log(1.0/999999.0) / x_75

	p_guess = np.ones(order_poly+2)
	p = least_squares(fun=residual, x0=p_guess, args=(m, spn, x, y))
	p_solved = p.x

	y_baseline = poly(p_solved[:-1], data[:,0]) 
	y_fit = y_baseline - logistic(p_solved[-1], m, spn, data[:,0])

	y_corrected = data[:,1] - y_baseline
	edge_step = p_solved[-1]

	if plot == True:
		plt.plot(x, y, '.', markersize = 2.0)
		plt.plot(data[:,0], y_fit, linewidth = 1.0)
		plt.plot(data[:,0], y_baseline, linewidth = 1.0)
		plt.plot(data[:,0], y_corrected/(-p_solved[-1]))

	return y_corrected/-p_solved[-1], edge_step

def spline_fitting(x, y, x_full, y_full):
	'''Spline fitting using spline interpolation implemented in SciPy.'''

	tck = interpolate.splrep(x, y, s = 0)
	y_fit = interpolate.splev(x_full, tck, der = 0)
	y_corrected = y_full - y_fit

	return y_fit, y_corrected

def polynomial_regression(x, y, x_full, y_full, order):
	'''Polynomial regression function, where x and y are used for fitting and x_full and y_full represent full
	data range.'''

	X_raw = np.tile(x, (order+1, 1))
	powers = np.arange(0, order+1)

	X = np.power(X_raw.T, powers)
	coef = np.dot(np.dot(inv(np.dot(X.T, X)), X.T), y)

	y_fit = poly(coef, x_full)
	y_corrected = y_full - y_fit
	
	return y_fit, y_corrected

def prep_data_fitting(data, se_array, normalize):
	'''Data preparation for pre-edge peak region baseline fitting. se_array is of length 4. se_array[0] and se_array[3]
	determine beginning and end of pre-edge peak region. se_array[1] and se_array[2] determine beginning and end
	of pre-edge peak signals.
	Function returns full pre-edge peak region datapoints as x_all and y_all.
	Baseline datapoints are returned as x and y.
	If normalize == True, beginning of pre-edge peak region is set to 0 (necessary for some exponential baseline functions)'''

	idx = find_nearest(data, se_array)

	x_f = data[:,0]
	y_f = data[:,1]

	x = np.r_[x_f[idx[0]:idx[1]], x_f[idx[2]:idx[3]]]
	y = np.r_[y_f[idx[0]:idx[1]], y_f[idx[2]:idx[3]]]

	data_all = data[idx[0]:idx[3]]
	x_all = data_all[:,0]
	y_all = data_all[:,1]

	if normalize == True:
		init = x_all[0]
		x = x - init
		x_all = x_all - init

	return x, y, x_all, y_all

def baseline_fitting_generic(data, se_array, function, normalize, p_guess = None, order_poly = 1, plot = False):
	'''Pre-Edge Peak Baseline Correction using either spline_fitting, polynomial_regression (using order_poly) 
	or any other defined function via least_squares (using p_guess).

	se_array provides information on pre-edge peak region beginning, end and signal beginning and end (details see prep_data_fitting function)
	function determines the function used for baseline correction.

	p_guess is used for any functions fitted using least_squares. length of p_guess determines polynomial order or 

	number of voigt peaks, if either of those two functions are used.

	order_poly is used for polynomial regression.'''

	x, y, x_f, y_f = prep_data_fitting(data, se_array, normalize)

	if function == spline_fitting:
		y_baseline, y_corrected = spline_fitting(x, y, x_f, y_f)

	elif function == polynomial_regression:
		y_baseline, y_corrected = polynomial_regression(x, y, x_f, y_f, order_poly)

	else:
		bounds = np.zeros((2, len(p_guess)))
		bounds[0] = -np.inf
		bounds[1] = np.inf

		if function == voigt:
			bounds[0] = 0.0
			bounds[1][0] = 1.0

		p = least_squares(fun=residual_generic, x0=p_guess, args=(x, y, function), bounds = bounds)
		p_solved = p.x

		y_baseline = function(p_solved, x_f)
		y_corrected = y_f - y_baseline

	data_corr = np.c_[x_f, y_corrected]

	if plot == True:
		plt.plot(x, y, '.', markersize = 2.0)
		plt.plot(x_f, y_baseline, linewidth = 1.0)
		plt.plot(x_f, y_corrected)

	return data_corr

def peak_fitting(data, peak_guess, plot = False):
	'''Fitting of pre-edge peaks using Voigt functions. Length of peak_guess determines number of Voigt functions used.
	Length of peak_guess has to be a multiple of four.
	For a given peak_guess sequence of four, peak_guess[0] is the Gaussian/Lorentzian ratio, peak_guess[1] is height, peak_guess[2] position
	and peak_guess[3] width.
	For Gaussian/Lorentzian ratio, there are bounds from 0 to 1.'''

	x = data[:,0]
	y = data[:,1]

	bounds = np.zeros((2, len(peak_guess)))
	bounds[1] = np.inf
	bounds[1][0::4] = 1
	
	p = least_squares(fun=residual_generic, x0=peak_guess, args=(x, y, voigt_mult), bounds = bounds)
	p_solved = p.x
	p_mat = np.reshape(p_solved, (-1, 4))

	x_full = np.arange(x[0], x[-1], 0.01)
	y_fit = voigt_mult(p_solved, x)
	resi = y - y_fit

	if plot == True:
		for i in p_mat[:,]:
			plt.plot(x_full, voigt(i, x_full))
		plt.plot(x, y, 'o-')
		plt.plot(x_full, voigt_mult(p_solved, x_full))
		plt.plot(x, resi, linewidth = 0.7)

	return p_solved

def wrapper(xd, stage):
	'''Wrapper function for XAS analysis. xd is a dictionary containing the analysis parameter, it can be constructed in stages.

	Stage 0 imports the data using the supplied filename and performs larch_baseline correction.

	Stage 1 performs LBC on the data, therefore, start, end and order_poly for LBC have to be provided.

	Stage 2 performs pre-edge peak region baseline correction, therefore, the parameters of baseline_fitting_generic 
	function have to be provided at this point.

	Stage 3 perform pre-edge peak fitting, necessitating the parameters for peak_fitting function.

	The wrapper function is to be used by increasing the stage index and then running the script. For each stage, the 
	corresponding results and plotted and the necessary parameters for the next stage can be decided by the user.''' 

	data = import_xdi(xd['name'])

	plotting = [False, False, False, False]
	plotting[stage] = True

	if stage >= 0:
		data_arr = larch_baseline(data, plot = plotting[0])

	if stage >= 1:
		mu_cor, edge_step_el = elevator_function_fitting(data_arr, xd['start_el'], xd['end_el'], xd['order_poly'], plot = plotting[1])

	if stage >= 2:
		norm_arr = np.c_[data.energy, data.norm]
		elev_arr = np.c_[data.energy, mu_cor]

		norm_cor = baseline_fitting_generic(norm_arr, xd['se_array'], xd['function'], True, xd['guess'], xd['order_poly_pp'], plot = plotting[2])
		elev_cor = baseline_fitting_generic(elev_arr, xd['se_array'], xd['function'], True, xd['guess'], xd['order_poly_pp'], plot = plotting[2])

	if stage >= 3:
		peaks_norm = peak_fitting(norm_cor, xd['peak_guess'], plot = plotting[3])
		peaks_elev = peak_fitting(elev_cor, xd['peak_guess'], plot = plotting[3])

		print(peaks_elev/peaks_norm * 100)

def main():
	'''Dictionary provides analysis paramters for different X-ray absorption spectra and the different stages
	can be visualized by changing stage index in wrapper function call.'''

	ap = {'cu_rt01' : {'name': 'cu_rt01.xmu', 'start_el': 8949., 'end_el': 9349., 'order_poly': 3, 
				  'se_array': np.array([8969., 8977., 8989., 8993.]), 'function': polynomial_regression, 
				  'guess': np.ones(6), 'order_poly_pp': 6, 'peak_guess': np.array([0.5, 0.3, 13., 2., 0.5, 0.1, 17., 2., 0.5, 0.2, 14., 2.])},

		  'fe3c_rt' : {'name': 'fe3c_rt.xdi', 'start_el': 7088., 'end_el': 7400., 'order_poly': 3, 
				  'se_array': np.array([7090., 7109., 7119., 7123.]), 'function': poly_expo, 
				  'guess': np.ones(7), 'order_poly_pp': 3, 'peak_guess': np.array([0.5, 0.1, 20., 2., 0.5, 0.05, 23.7, 2., 0.5, 0.06, 20.5, 2.])},

		  'fe2o3_rt1' : {'name': 'fe2o3_rt1.xmu', 'start_el': 7082., 'end_el': 7417., 'order_poly': 3, 
				  'se_array': np.array([7100., 7108., 7120.,7127.]), 'function': voigt, 
				  'guess': np.array([0.5, 0.6, 35., 3.0]), 'order_poly_pp': 3, 'peak_guess': np.array([0.5, 0.3, 13., 2., 0.5, 0.1, 17., 2.])},

		  'FeCl3' : {'name': 'FeCl3_XAS.txt', 'start_el': 7079., 'end_el': 7400., 'order_poly': 3,
		  		  'se_array': np.array([7086., 7107., 7119., 7123.]), 'function': poly_expo,
		  		  'guess': np.ones(5), 'order_poly_pp': 3, 'peak_guess': np.array([0.5, 0.025, 28., 2., 0.5, 0.007, 31.3, 2., 0.5, 0.02, 29., 2.])}}


	wrapper(ap['cu_rt01'], 3)
	plt.show()

if __name__ == '__main__':
	main()