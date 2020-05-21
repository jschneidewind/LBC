import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import find_peaks
from scipy.optimize import least_squares

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

def import_data():
	'''Importing already converted mofoil_190 data file from Farrel Lytle database and converting y_axis to mu'''

	data = np.loadtxt('mofoil_190_converted.txt')
	return np.c_[data[:,0], data[:,3]/data[:,1]]

def white_line_linreg(data, peak, linreg_start_offset, linreg_interval_len):

	linreg_start = data[:,0][peak] + linreg_start_offset
	linreg_idx = find_nearest(data[:,0], (linreg_start, linreg_start + linreg_interval_len))

	data_linreg = data[linreg_idx[0]:linreg_idx[1]]
	slope, intercept, _, _, _, = linregress(data_linreg[:,0], data_linreg[:,1])

	return slope, intercept

def white_line_integration(data, background, peak, integration_width):

	integration_idx = find_nearest(data[:,0], (data[:,0][peak] - integration_width, data[:,0][peak] + integration_width))
	
	data_int = data[integration_idx[0]:integration_idx[1]]
	background_int = background[integration_idx[0]:integration_idx[1]]

	data_area = np.trapz(data_int[:,1], data_int[:,0])
	background_area = np.trapz(background_int, data_int[:,0])

	return data_area - background_area

def white_line_intensity(data, background, peaks, integration_width = 20., norm_start = 50., norm_interval_len = 50.):
	'''Analysis of white line intensities adapted from Okamato 2005 (see Pearson_analysis) with corresponding presets for 
	integration width, norm_start and norm_interval_len;
	integration_width (in eV) determines width of window over which white lines are integrated)
	norm_start (in eV) determines the starting point for the normalization interval, relative to the position of the L3 white line maximum;
	norm_interval (in eV) determines the length  of the normalization interval.'''

	white_line_area = []

	for peak in peaks:
		area = white_line_integration(data, background, peak, integration_width)
		white_line_area.append(area)
	white_line_area = np.asarray(white_line_area)

	norm_start = data[:,0][peaks[0]] + norm_start
	norm_idx = find_nearest(data[:,0], (norm_start, norm_start + norm_interval_len))
	norm_x = data[:,0][norm_idx[0]:norm_idx[1]]
	norm_data = background[norm_idx[0]:norm_idx[1]]

	norm_area = np.trapz(norm_data, norm_x)

	return np.sum(white_line_area/norm_area)

def Pearson_analysis(data, prominence = 0.1, linreg_start_offset = 20., linreg_interval_len = 50.):
	'''Pearson analysis for white line intensities as described in 'EELS Analysis of the Electronic Structure and Microstructure of Metals', Okamato, 2005;
	prominence determines cut-off for white line peak identification;
	linreg_start_offset (in eV) determines starting point of linear regression succeeding each white line relative to white line maximum;
	linreg_interval_len (in eV) determines the length of the interval used for the white line linear regression'''

	peaks = find_peaks(data[:,1], prominence = prominence)[0]

	linreg =[]

	for peak in peaks:
		slope, intercept = white_line_linreg(data, peak, linreg_start_offset, linreg_interval_len)
		linreg.append((slope, intercept))
	linreg = np.asarray(linreg)

	background = linreg[0][0] * data[:,0] + linreg[0][1]
	background[:peaks[0]] = 0
	background[peaks[1]:] = linreg[1][0] * data[:,0][peaks[1]:] + linreg[1][1]

	plt.plot(data[:,0], background)

	white_line_int = white_line_intensity(data, background, peaks)

	return white_line_int

def poly(a, x):

	y = a[0] * x**0

	for i in range(1, len(a)):
		y += a[i] * x**i

	return y

def logistic(c, m, spn, x):

	return (c/(1 + np.exp(spn*(x - m))))

def residual_multi_step(p, m_spn, x, y):

	y_poly = poly(p[:-len(m_spn)], x)
	res = y - y_poly

	for counter, i in enumerate(m_spn):
		res -= logistic(p[-(counter+1)], i[0], i[1], x)

	return res

def calc_m_spn(data, idx):
	'''Calculation of sigmoidal midpoint and logistic growth rate for LBC'''

	m = (data[:,0][idx[0]] + data[:,0][idx[1]]) / 2
	x_75 = (data[:,0][idx[1]] - m) / 2
	spn = spn = np.log(1.0/9.99e18) / x_75 #9.99e18 divisor ensures high steepness, comparable with step function

	return m, spn

def multi_step_LBC(data, feature_width, order_poly, prominence = 0.1, plotting = False):
	'''Perfoming LBC with multiple logistic function to compensate for multiple baseline-
	varying features in the data;
	number of features are determined automatically by peak picking based on prominence criteria;
	feature width (in eV) controls width of feature windows which are not considered during baseline fitting;
	order_poly controls order of polynomial used for baseline fitting'''

	peaks = find_peaks(data[:,1], prominence = prominence)[0]

	idx_list = []

	for peak in peaks:
		idx = find_nearest(data[:,0], (data[:,0][peak] - feature_width, data[:,0][peak] + feature_width))
		idx_list.append(idx)
	idx_list = np.asarray(idx_list)

	idx_comp = np.ravel(idx_list)
	idx_comp = np.append(idx_comp, None)
	idx_comp = np.insert(idx_comp, 0, None)
	idx_comp = np.reshape(idx_comp, (np.int(len(idx_comp)/2), 2))

	data_baseline = data[idx_comp[0][0]:idx_comp[0][1]]

	for i in idx_comp[1:]:
		data_baseline = np.append(data_baseline, data[i[0]:i[1]], axis = 0)

	m_spn = []
	for i in idx_list:
		m, spn = calc_m_spn(data, i)
		m_spn.append((m, spn))
	m_spn = np.asarray(m_spn)

	p_guess = np.ones(order_poly + len(m_spn) + 1)
	p = least_squares(fun=residual_multi_step, x0=p_guess, args=(m_spn, data_baseline[:,0], data_baseline[:,1]))

	y_baseline = poly(p.x[:-len(m_spn)], data[:,0])

	y_corrected = data[:,1] - y_baseline
	y_fit = np.copy(y_baseline)

	for counter, i in enumerate(m_spn):
		y_fit += logistic(p.x[-(counter+1)], i[0], i[1], data[:,0])

	if plotting == True:
		plt.plot(data[:,0], data[:,1])
		plt.plot(data[:,0], y_baseline)
		plt.plot(data[:,0], y_fit)
		plt.plot(data_baseline[:,0], data_baseline[:,1], '.')

	white_line_int = white_line_intensity(data, y_fit, peaks)

	return white_line_int

def main():
	data = import_data()
	print(Pearson_analysis(data))
	print(multi_step_LBC(data, 30., 4, plotting = True))
	plt.show()

if __name__ == '__main__':
	main()