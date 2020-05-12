import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import least_squares
from scipy.stats import linregress
from timeit import default_timer as timer
from scipy.stats import pearsonr
from scipy.stats import t
from scipy.stats.mstats import gmean

#np.set_printoptions(suppress=True)

#### qPCR Expo Analysis Functions ####

def convert_qPCR_data_short(data, x, row):

	return np.c_[x, data[row,:]]

def poly(a, x):

	y = a[0] * x**0

	for i in range(1, len(a)):
		y += a[i] * x**i

	return y

def logistic(c, m, spn, x):

	return (c/(1 + np.exp(spn*(x - m))))

def expo(p, x):

	return p[0] * p[1]**x

def expo_noe(p, E, x):

	return p * E**x

def residual_expo_noe(p, E, x, y):

	y_expo = expo_noe(p, E, x)
	res = y - y_expo

	return res

def residual_expo(p, x, y):

	y_expo = expo(p, x)
	res = y - y_expo

	return res

def residual(p, m, spn, x, y, idx_s, weight_factor):

	y_poly = poly(p[:-1], x)
	y_log = logistic(p[-1], m, spn, x)

	weight = np.ones(len(x))
	weight[:(idx_s-2)] = weight_factor

	res = (y + y_log - y_poly) * weight

	return res

def residual_pre_signal(p, x, y):

	y_poly = poly(p, x)
	res = y - y_poly

	return res

def consecutive(data):

	return np.split(data, np.where(np.diff(data) != 1)[0]+1)

def find_start_end(data, n_dp, n_std):
	'''Determine signal start and end point by fitting linear regression to first and last n_dp datapoints (omitting first two)

	Signal start point is the first point exceeding the start linear regression model by more than n_std standard deviations and being
	only followed by datapoints also exceeding that threshold

	Signal end point is the last point more than n_std standard deviations below the end linear regression model and only being
	preceded by datapoints also below that threshold.'''

	m_start, c_start, _, _, _ = linregress(data[:,0][2:n_dp+2], data[:,1][2:n_dp+2])
	m_end, c_end, _, _, _ = linregress(data[:,0][-n_dp:], data[:,1][-n_dp:])

	lin_start = m_start * data[:,0] + c_start
	lin_end = m_end * data[:,0] + c_end

	diff_start = data[:,1] - lin_start
	diff_end = data[:,1] - lin_end

	std_start = np.std(data[:,1][2:n_dp+2])
	std_end = np.std(data[:,1][-n_dp:])

	thres_start = np.where(diff_start > (std_start * n_std))[0]
	thres_end = np.where(diff_end < (-std_end * n_std))[0]

	start = consecutive(thres_start)[-1][0]
	end = consecutive(thres_end)[0][-1]

	return np.array([start, end])

def elevator_function_fitting(data, start, end, order_poly, weight_factor, plotting = False):
	'''LBC function'''

	idx = np.array([start, end], dtype=int)
	x = np.r_[data[:,0][2:idx[0]], data[:,0][idx[1]:]]
	y = np.r_[data[:,1][2:idx[0]], data[:,1][idx[1]:]]

	m = (data[:,0][idx[0]] + data[:,0][idx[1]]) / 2
	x_75 = (data[:,0][idx[1]] - m) / 2
	spn = np.log(1.0/999999.0) / x_75

	p_guess = np.ones(order_poly+2)
	p = least_squares(fun=residual, x0=p_guess, args=(m, spn, x, y, idx[0], weight_factor))
	p_solved = p.x

	y_baseline = poly(p_solved[:-1], data[:,0]) 
	y_corrected = data[:,1] - y_baseline
	data_cor = np.c_[data[:,0], y_corrected]

	if plotting == True:
		plt.figure()
		plt.plot(data[:,0], y_baseline, linewidth = 1.0, color = 'blue')
		plt.plot(data[:,0], data[:,1], 'o-', markersize = 2., linewidth = 1., color = 'green')
		plt.plot(data[:,0], y_corrected, 'o-', markersize = 2., linewidth = 1., color = 'blue')

	return data_cor

def pre_signal_fitting(data, end, order_poly):
	'''Pre-signal fitting baseline correction function'''

	idx = np.array([end], dtype=int)
	x = data[:,0][:idx[0]]
	y = data[:,1][:idx[0]]

	p_guess = np.ones(order_poly+1)
	p = least_squares(fun=residual_pre_signal, x0=p_guess, args=(x, y))
	p_solved = p.x

	y_baseline = poly(p_solved, data[:,0])
	y_corrected = data[:,1] - y_baseline

	return np.c_[data[:,0], y_corrected]

def prep_fit_expo(data, idx_s):
	'''Preparation of fitting of exponential amplifcation function. Exponential amplification interval is
	selected in the range of signal start to second derivative maximum (SDM itself being excluded from the interval)'''

	first_order_change = np.diff(data[:,1])
	second_order_change = np.diff(first_order_change)

	idx_e = np.argmax(second_order_change)
	idx = np.array([idx_s, idx_e])

	data_intv = data[idx[0]:idx[1]]

	x = data_intv[:,0]
	y = data_intv[:,1]

	return x, y

def fit_expo(data, idx_s):
	'''Fitting two paramter exponential function to obtain E'''

	x, y = prep_fit_expo(data, idx_s)

	p_guess = np.ones(2)

	p = least_squares(fun = residual_expo, x0 = p_guess, args = (x, y))
	p_solved = p.x

	return p_solved, x, y

def fit_expo_noe(x, y, E):
	'''Fitting one paramter exponential function to obtain target quantity F0'''

	p_guess = np.ones(1)

	p = least_squares(fun = residual_expo_noe, x0 = p_guess, args = (E, x, y))

	print(p.success)

	p_solved = p.x

	return p_solved

def determine_E(E_values):
	'''Determine E as the average of supplied E values but excluding those below 1.5 and above 2.5 from the 
	mean calculation'''

	E_raw = E_values[:,1][np.argsort(E_values[:,1])]
	range_idx = np.searchsorted(E_raw, [1.5, 2.5])
	E_filtered = E_raw[range_idx[0]:range_idx[1]]
	E = np.mean(E_filtered)

	return E

def fit_expo_determine_E(data_complete, standard_cycles, samples_E, no_dp, no_std, order_poly, order_poly_pre, weight_factor):
	'''PCR analysis algorithm. Data is a 2d array of different qPCR reactions for one gene.

	Function randomly selects samples_E reactions from array, for which baseline correction and two-paratmer exponential
	function fitting are performed. The obtained E values are averaged to obtain E for this gene.

	Afterwards, for all reactions baseline correction is performed and the baseline corrected signals are analyzed 
	by fitting the one paramter exponential function using the previously obtained E to obtain F0 (target quantity),
	which is returned'''

	cycles = np.arange(0, len(data_complete[0,:]))
	idx = np.arange(0, len(standard_cycles))

	random_idx = np.random.choice(idx, samples_E, replace = False)
	cycles_eval = standard_cycles[random_idx]
	cycles_other = np.delete(standard_cycles, random_idx)

	p_E_arr = []
	x_arr = []
	y_arr = []

	p_E_pre_arr = []
	x_pre_arr = []
	y_pre_arr = []

	for i in cycles_eval:

		data = convert_qPCR_data_short(data_complete, cycles, i)
		idx_se = find_start_end(data, no_dp, no_std)

		data_cor = elevator_function_fitting(data, start = data[:,0][idx_se[0]], end = data[:,0][idx_se[1]], order_poly = order_poly, weight_factor = weight_factor)
		expo_p, x, y = fit_expo(data_cor, idx_se[0])

		data_cor_pre = pre_signal_fitting(data, end = data[:,0][idx_se[0]], order_poly = order_poly_pre)
		expo_p_pre, x_pre, y_pre = fit_expo(data_cor_pre, idx_se[0])

		p_E_arr.append(expo_p)
		x_arr.append(x)
		y_arr.append(y)

		p_E_pre_arr.append(expo_p_pre)
		x_pre_arr.append(x_pre)
		y_pre_arr.append(y_pre)

	p_E = np.asarray(p_E_arr)
	x_np = np.asarray(x_arr)
	y_np = np.asarray(y_arr)

	p_E_pre = np.asarray(p_E_pre_arr)
	x_pre_np = np.asarray(x_pre_arr)
	y_pre_np = np.asarray(y_pre_arr)

	E = determine_E(p_E)
	E_pre = determine_E(p_E_pre)

	f0_arr = []
	f0_pre_arr = []

	for x_data, y_data, x_data_pre, y_data_pre in zip(x_np, y_np, x_pre_np, y_pre_np):

		expo_p_noe = fit_expo_noe(x_data, y_data, E)
		expo_p_noe_pre = fit_expo_noe(x_data_pre, y_data_pre, E_pre)

		f0_arr.append(expo_p_noe)
		f0_pre_arr.append(expo_p_noe_pre)


	for i in cycles_other:

		data = convert_qPCR_data_short(data_complete, cycles, i)
		idx_se = find_start_end(data, no_dp, no_std)

		data_cor = elevator_function_fitting(data, start = data[:,0][idx_se[0]], end = data[:,0][idx_se[1]], order_poly = order_poly, weight_factor = weight_factor)
		x_other, y_other = prep_fit_expo(data_cor, idx_se[0])
		expo_p_noe = fit_expo_noe(x_other, y_other, E)

		data_cor_pre = pre_signal_fitting(data, end = data[:,0][idx_se[0]], order_poly = order_poly_pre)
		x_other_pre, y_other_pre = prep_fit_expo(data_cor_pre, idx_se[0])
		expo_p_noe_pre = fit_expo_noe(x_other_pre, y_other_pre, E_pre)

		f0_arr.append(expo_p_noe)
		f0_pre_arr.append(expo_p_noe_pre)

	f0 = np.asarray(f0_arr)
	f0_pre = np.asarray(f0_pre_arr)

	cycles_all = np.r_[cycles_eval, cycles_other]
	idx_sort = np.flip(np.argsort(cycles_all), axis = 0)

	f0_sorted = f0[idx_sort]
	f0_sorted_pre = f0_pre[idx_sort]

	return f0_sorted, f0_sorted_pre, E, E_pre

def full_analysis(sample_E, no_dp, no_std, order_poly, order_poly_pre, weight_factor):
	'''Function performs full analysis of Vermeulen technical dataset. One gene from Data_Vermeulen_A is excluded (as
	was done in Ruijter et al. (Methods, 2013))

	Returns array of f0 values for all reactions of all considered genes obtained using LBC and pre-signal fitting for
	baseline correction'''

	file_id = ['A', 'B', 'C', 'D']

	sheet_names = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
	sheets = np.arange(0, 16)
	sheets_clean = np.delete(sheets, 4)

	standard_cycles = np.asarray([325, 327, 329, 331, 333, 335, 361, 363, 365, 367, 369, 371, 373, 375, 377])

	f0_np = np.zeros(15)
	f0_np_pre = np.zeros(15)

	df_A = pd.read_excel('Data_Vermeulen_A.xls', sheet_name = sheet_names, header = 0)

	for i in sheets_clean:

		print('Data: A', 'sheet:', i)

		df_clean = df_A[i].drop(['Well', 'Sample'], axis=1)
		data = df_clean.to_numpy()

		f0_result, f0_result_pre, E, E_pre = fit_expo_determine_E(data, standard_cycles, sample_E, no_dp, no_std, order_poly, order_poly_pre, weight_factor)
		
		f0_np = np.c_[f0_np, f0_result]
		f0_np_pre = np.c_[f0_np_pre, f0_result_pre]


	for i in file_id[1:]:

		name = 'Data_Vermeulen_%s.xls' % i
		df = pd.read_excel(name, sheet_name = sheet_names, header = 0)

		for j in sheets:

			print('Data:', i, 'sheet:', j)

			df_clean = df[j].drop(['Well', 'Sample'], axis=1)
			data = df_clean.to_numpy()

			f0_result, f0_result_pre, E, E_pre = fit_expo_determine_E(data, standard_cycles, sample_E, no_dp, no_std, order_poly, order_poly_pre, weight_factor)
			
			f0_np = np.c_[f0_np, f0_result]
			f0_np_pre = np.c_[f0_np_pre, f0_result_pre]

	return np.delete(f0_np, 0, axis = 1), np.delete(f0_np_pre, 0, axis = 1)


#### qPCR Analysis Method Performance Functions ####

def import_target_quant(name):

	df = pd.read_excel(name, header = None)
	df_clean = df.drop(2, axis = 1)

	data = df_clean.to_numpy()

	return data

ms_Cq = import_target_quant('MS_within_Cq_Biomarker.xlsx')[0]

def calc_steyx(x, y):

	X = sm.add_constant(x)
	model = sm.OLS(y, X)
	fit = model.fit()

	return np.sqrt(fit.mse_resid)

def generate_intv():

	space = np.linspace(0, 15, 6)
	a = space[1:]
	b = space[:-1]
	
	return np.c_[b, a].astype(int)

def bias_deviat_analysis(data):
	'''First part of performance indicator calculations adapted from Ruijter et al. (Methods, 2013)'''

	conc = np.repeat([15 * np.geomspace(1, 10000, num=5)], 3)
	
	high_conc_mean = np.mean(conc[12:])
	high_conc_data_mean = np.mean(data[12:], axis = 0)

	conc_norm = conc / high_conc_mean
	data_norm = data / high_conc_data_mean

	mean_low = np.mean(data_norm[:3], axis = 0)
	mean_high = np.mean(data_norm[12:], axis = 0)

	bias = mean_high / mean_low

	conc_log = np.log10(conc_norm)
	data_log = np.log10(data_norm)
	data_raw_log = np.log10(data)

	slope_arr = []
	rvalue_arr = []
	stexy_arr = []

	for i in range(len(data_log[0,:])):

		slope, intercept, rvalue, pvalue, stderr = linregress(conc_log, data_log[:,i])
		stexy = calc_steyx(conc_log, data_log[:,i])

		slope_arr.append(slope)
		rvalue_arr.append(rvalue)
		stexy_arr.append(stexy)

	slope = np.asarray(slope_arr)
	rvalue = np.asarray(rvalue_arr)
	stexy = np.asarray(stexy_arr)

	var = np.var(data_log, axis = 0, ddof = 1)
	ss_total = var * 14

	intv = generate_intv()

	var_intv_arr = []
	ss_dilution_arr = []

	for i in intv:
		var_intv = np.var(data_log[i[0]:i[1]], axis = 0, ddof = 1)
		var_intv_arr.append(var_intv)

		ss_i = np.var(data_raw_log[i[0]:i[1]], axis = 0, ddof = 1) * 2
		ss_dilution_arr.append(ss_i)

	var_intv = np.asarray(var_intv_arr)
	ss_dilution = np.asarray(ss_dilution_arr)

	ss_intv = var_intv * 2
	ss_within = np.sum(ss_intv, axis = 0)

	ms_within = np.sum(ss_dilution, axis = 0) / 10

	ss_between = ss_total - ss_within

	ss_res = stexy ** 2 * 13
	ss_regression = ss_total - ss_res
	ss_deviation = ss_res - ss_within

	ms_between = ss_between / 4

	linearity = ss_deviation / 3
	reproducibility = ss_within / 10

	ratio = ms_within / ms_Cq
	ratio_min = ratio[np.argmin(ratio)]
	ratio_max = ratio[np.argmax(ratio)]

	return bias, slope, rvalue, linearity, reproducibility, ratio, ratio_min, ratio_max

def calc_detectable_difference(data):
	'''Second part of performance indicator calculations adapted from Ruijter et al. (Methods, 2013)'''

	conc = np.repeat([15 * np.geomspace(1, 10000, num=5)], 3)

	conc_log = np.log10(conc)
	data_log = np.log10(data)

	steyx_arr = []

	for i in range(len(data_log[0,:])):
		steyx_i = calc_steyx(conc_log, data_log[:,i])
		steyx_arr.append(steyx_i)

	steyx  = np.asarray(steyx_arr)

	intv = generate_intv()
	mean_arr = []

	for i in intv:
		mean_i = np.mean(data_log[i[0]:i[1]], axis = 0)
		mean_arr.append(mean_i)

	mean_intv = np.asarray(mean_arr)

	mean_x = np.mean(conc_log)
	ss_x = np.var(conc_log, ddof = 1) * 13

	conc_log_mean = conc_log[[0, 3, 6, 9, 12]]
	sq_part = np.sqrt(1./3. + (((conc_log_mean - mean_x)**2)/ss_x))
	se_yfit = np.outer(sq_part, steyx)

	t_intv = t.ppf(1-0.0125, 2)

	ci_y_upper = mean_intv + t_intv * se_yfit
	ci_y_lower = mean_intv - t_intv * se_yfit

	ci_y_upper_no_log = 10 ** ci_y_upper
	ci_y_lower_no_log = 10 ** ci_y_lower
	mean_intv_no_log = 10 ** mean_intv

	fold_up = ci_y_upper_no_log / mean_intv_no_log
	fold_down = mean_intv_no_log / ci_y_lower_no_log

	detectable_difference = gmean(fold_up, axis = 0)

	return detectable_difference

def wrapper(data):
	'''Wrapper function for performance indicator calculation returning array with performance indicators for all genes'''

	bias, slope, rvalue, linearity, reproducibility, ratio, ratio_min, ratio_max = bias_deviat_analysis(data)
	detectable_difference = calc_detectable_difference(data)

	result = np.c_[bias, slope, rvalue, linearity, reproducibility, ratio, detectable_difference].T

	return result, ratio_min, ratio_max

def get_performance(sample_E, no_dp, no_std, order_poly, order_poly_pre, weight_factor):
	'''Calculation of performance indicators for qPCR methods supplied in Target_Quant_Dilution_Biomarker.xlsx 
	and analysis of qPCR data using LBC and pre-signal fitting and calculation of corresponding performance
	indicators'''

	sn = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	df = pd.read_excel('Target_Quant_Dilution_Biomarker.xlsx', sheet_name = sn, header = None)

	performance_arr = []

	for i in range(10):

		df_clean = df[i].drop(2, axis = 1)
		data = df_clean.to_numpy()
	
		result, _, _ = wrapper(data)
		performance_arr.append(result)

	f0_np, f0_np_pre = full_analysis(sample_E, no_dp, no_std, order_poly, order_poly_pre, weight_factor)

	result, _, _ = wrapper(f0_np)
	result_pre, _, _ = wrapper(f0_np_pre)

	performance_arr.append(result)
	performance_arr.append(result_pre)

	performance = np.asarray(performance_arr)

	return performance

def sort_plot_performance(sample_E, no_dp, no_std, order_poly, order_poly_pre, weight_factor):
	'''Plotting of performance indicators. Plots are sorted for visualization based on median value of
	respective performance indicator.'''

	performance = get_performance(sample_E, no_dp, no_std, order_poly, order_poly_pre, weight_factor)

	method = ['LinRegPCR', '5PSM', 'FPK', 'LRE-EMax', 'LRE-E100', 'Cy0', 'MAK2', 'DART', 'FPLM', 'PCR-Miner', 'LBC', 'PSF']
	indicator_label = ['bias', 'slope', 'rvalue', 'linearity', 'reproducibility', 'ratio', 'detectable_difference']

	for i in range(len(performance[0, :, 0])):

		median = np.median(performance[:, i, :], axis = 1)
		sort_idx = np.argsort(median)
		performance[:, i, :] = performance[:, i, :][sort_idx]
		method_sorted = np.asarray(method)[sort_idx]

		performance_indicator = performance[:, i, :].T

		plt.figure()
		plt.title(indicator_label[i])
		plt.boxplot(performance_indicator, whis = [5, 95], showfliers = False, patch_artist = True, labels = method_sorted)		

def get_performance_El_Pre(sample_E, no_dp, no_std, order_poly, order_poly_pre, weight_factor):
	'''qPCR analysis using LBC and pre-signal fitting with subsequent calculation of performance indicators 
	(excluding other methods)'''

	performance_arr = []

	f0_np, f0_np_pre = full_analysis(sample_E, no_dp, no_std, order_poly, order_poly_pre, weight_factor)

	result, _, _ = wrapper(f0_np)
	result_pre, _, _ = wrapper(f0_np_pre)

	performance_arr.append(result)
	performance_arr.append(result_pre)

	performance = np.asarray(performance_arr)

	return performance

def sort_plot_performance_El_Pre(sample_E, no_dp, no_std, order_poly, order_poly_pre, weight_factor):
	'''Plotting of LBC and pre-signal fitting performance indicators'''

	performance = get_performance_El_Pre(sample_E, no_dp, no_std, order_poly, order_poly_pre, weight_factor)

	method = ['LBC', 'Pre-Signal Fitting']
	indicator_label = ['bias', 'slope', 'rvalue', 'linearity', 'reproducibility', 'ratio', 'detectable_difference']

	for i in range(len(performance[0, :, 0])):

		median = np.median(performance[:, i, :], axis = 1)
		sort_idx = np.argsort(median)
		performance[:, i, :] = performance[:, i, :][sort_idx]
	
		method_sorted = np.asarray(method)[sort_idx]
		performance_indicator = performance[:, i, :].T

		plt.figure()
		plt.title(indicator_label[i])
		plt.boxplot(performance_indicator, whis = [5, 95], showfliers = False, patch_artist = True, labels = method_sorted)		

def analyse_single_sheet(file_name, sheet, samples_E, no_dp, no_std, order_poly, order_poly_pre, weight_factor):
	'''Function to analyze single sheet of Vermeulen technical dataset.'''

	df = pd.read_excel(file_name, sheet_name = sheet, header = 0)
	df_clean = df.drop(['Well', 'Sample'], axis=1)
	data_raw = df_clean.to_numpy()

	standard_cycles = np.asarray([325, 327, 329, 331, 333, 335, 361, 363, 365, 367, 369, 371, 373, 375, 377])

	f0_result, f0_result_pre, E, E_pre = fit_expo_determine_E(data_raw, standard_cycles, samples_E, no_dp, no_std, order_poly, order_poly_pre, weight_factor)

	return E, E_pre

def analyse_single_row(file_name, sheet, row, no_dp, no_std, order_poly, order_poly_pre, E, E_pre, weight_factor):
	'''Function to analyze single reaction from Vermeulen technial dataset.'''

	df = pd.read_excel(file_name, sheet_name = sheet, header = 0)
	df_clean = df.drop(['Well', 'Sample'], axis=1)
	data_raw = df_clean.to_numpy()

	cycles = np.arange(0, len(data_raw[0,:]))

	data = convert_qPCR_data_short(data_raw, cycles, row)
	idx_se = find_start_end(data, no_dp, no_std)

	data_cor = elevator_function_fitting(data, start = data[:,0][idx_se[0]], end = data[:,0][idx_se[1]], order_poly = order_poly, weight_factor = weight_factor, plotting = True)
	x_other, y_other = prep_fit_expo(data_cor, idx_se[0])
	expo_p_noe = fit_expo_noe(x_other, y_other, E)

	data_cor_pre = pre_signal_fitting(data, end = data[:,0][idx_se[0]], order_poly = order_poly_pre)
	x_other_pre, y_other_pre = prep_fit_expo(data_cor_pre, idx_se[0])
	expo_p_noe_pre = fit_expo_noe(x_other_pre, y_other_pre, E_pre)

def main():
	'''Run sort_plot_performance to to qPCR analysis with LBC and PSL (pre-signal fitting) and plot comparison with other qPCR analysis tools.
	Run sort_plot performance to solely compare LBC and PSL

	The used indicator names correspond to the names used in Ruijter et al. (Methods, 2013) in the followign way:
	   Increased Variation = Ratio, Resolution = Detectable Difference, Bias = Bias, Linearity = Linearity, Precision = Reproducbility

	For paramters, use e.g.: sample_E = 10, no_dp = 5, no_std = 7, order_poly = 2, order_poly_pre = 1, weight_factor = 8 (high number of baseline datpoints regime)
							 sample_E = 10, no_dp = 3, no_std = 5, order_poly = 2, order_poly_pre = 1, weight_factor = 8 (low number of baseline datpoints regime)

	The weight factor only applies to LBC (weighing pre-signal datapoints in optimization), order_poly is the order of the baseline polynomial used for LBC while order_poly_pre is the order of the baseline polynomial
	used for PSL.

	Sample_E is the number reactions for one gene which are sampled to obtain amplification efficiency E (maximum 15)

	no_dp is the number of datapoints used for the linear regression models used in the determination of signal start and end point
	no_std is the number of standard deviations used as a threshold to determine the signal start and end points'''

	start = timer ()
	sort_plot_performance_El_Pre(10, 5, 7, 2, 1, 8.)
	end = timer()
	print('Time:', end - start)  # runs for ca. 1 min on a quad-core MBP (2014) for sample_E = 10

	E, E_pre = analyse_single_sheet('Data_Vermeulen_B.xls', 1, 5, 5, 7, 2, 1, 8.)
	analyse_single_row('Data_Vermeulen_B.xls', 1, 377, 5, 7, 2, 1, E, E_pre, 8.)	

	plt.show()

if __name__ == '__main__':
	main()