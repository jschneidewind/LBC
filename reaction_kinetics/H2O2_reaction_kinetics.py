import numpy as np
from numpy.linalg import inv
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

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

def import_txt(name, channel, sensors):

    raw_data = np.genfromtxt(name, skip_header=14+(2*sensors-2), encoding='ISO8859', usecols = (0,1,2,3,4,5,6,7,8,9,10,11))
    data = raw_data[:,[2, 2+channel, 6+channel]]
    return data

def poly(a, x):

    y = a[0] * x**0

    for i in range(1, len(a)):
        y += a[i] * x**i

    return y

def logistic(c, m, spn, x):

    return (c/(1 + np.exp(spn*(x - m))))

def first_order(p, t):

    return 1 - (p[0] * np.exp(-p[1]*t))

def first_order_combined(p, t):

    return p[0] - (p[0] * np.exp(-p[1]*t))

def first_order_shift(p, t):
    '''Function with value zero for t < p[2], first order for t > p[2]'''

    idx = find_nearest(t, p[2])[0]

    t_base = t[:idx+1]
    t_feature = t[idx+1:]

    y_base = t_base * 0
    y_feature = first_order_combined(p[:2], (t_feature - p[2]))

    return np.r_[y_base, y_feature]

def residual(p, m, spn, x, y):

    y_poly = poly(p[:-1], x)
    y_log = logistic(p[-1], m, spn, x)
    res = y - y_log - y_poly

    return res

def residual_generic(p, x, y, function):

    y_fit = function(p, x)
    res = y - y_fit

    return res

def polynomial_regression(x, y, order):
    '''Polynomial regression function, fitting polynomial of order 'order' to x and y using ordinary least squares''' 

    X_raw = np.tile(x, (order+1, 1))
    powers = np.arange(0, order+1)
    X = np.power(X_raw.T, powers)

    coef = np.dot(np.dot(inv(np.dot(X.T, X)), X.T), y)

    return coef

def elevator_function_fitting(data, start, end, order_poly, plotting = False, return_full = True, p_guess = None):
    '''LBC function'''

    idx = find_nearest(data, (start, end))
    x = np.r_[data[:,0][:idx[0]], data[:,0][idx[1]:]]
    y = np.r_[data[:,1][:idx[0]], data[:,1][idx[1]:]]

    m = (data[:,0][idx[0]] + data[:,0][idx[1]]) / 2
    x_75 = (data[:,0][idx[1]] - m) / 2
    spn = np.log(1.0/999999.0) / x_75

    if p_guess is None:
        p_guess = np.ones(order_poly+2)

    p = least_squares(fun=residual, x0=p_guess, args=(m, spn, x, y))
    p_solved = p.x

    y_baseline = poly(p_solved[:-1], data[:,0]) 
    y_fit = y_baseline + logistic(p_solved[-1], m, spn, data[:,0])
    y_corrected = data[:,1] - y_baseline
    data_corr = np.c_[data[:,0], y_corrected]

    res = residual(p_solved, m, spn, x, y)
    res = np.sum(res**2)

    if plotting == True:
        plt.plot(x, y, '.', markersize = 2.0)
        plt.plot(data[:,0], y_fit, linewidth = 1.0)
        plt.plot(data[:,0], y_baseline, linewidth = 1.0)
        plt.plot(data[:,0], y_corrected)

    if return_full == False:
        return res 

    else:
        return data_corr, y_baseline, p_solved[-1], p_solved

def pre_signal_fitting(data, end, order_poly, plotting = False):
    '''Pre-signal fitting for baseline correction'''

    idx = find_nearest(data, end)
    x = data[:,0][:idx[0]]
    y = data[:,1][:idx[0]]

    p_guess = np.ones(order_poly+1)
    p = least_squares(fun=residual_generic, x0=p_guess, args=(x, y, poly))
    p_solved = p.x

    y_baseline = poly(p_solved, data[:,0])
    y_corrected = data[:,1] - y_baseline 

    if plotting == True:
        plt.plot(x, y, '.', markersize = 2.0)
        plt.plot(data[:,0], y_baseline, linewidth = 1.0)
        plt.plot(data[:,0], y_corrected)

    data_corr = np.c_[data[:,0], y_corrected]

    return data_corr 

def post_signal_fitting(data, end, order_poly, plotting = False):
    '''Post-signal fitting for baseline correction'''

    idx = find_nearest(data, end)
    x = data[:,0][idx[0]:]
    y = data[:,1][idx[0]:]

    p_guess = np.ones(order_poly+1)
    p = least_squares(fun=residual_generic, x0=p_guess, args=(x, y, poly))
    p_solved = p.x

    y_baseline = poly(p_solved, data[:,0])
    y_corrected = data[:,1] - y_baseline 

    if plotting == True:
        plt.plot(x, y, '.', markersize = 2.0)
        plt.plot(data[:,0], y_baseline, linewidth = 1.0)
        plt.plot(data[:,0], y_corrected)

    data_corr = np.c_[data[:,0], y_corrected-np.mean(y_corrected[:10])]

    return data_corr

def fitting_prep(data, start, end, final_conc):
    '''Normalization of baseline corrected data for fitting of integrated, normalized first-order rate law.
    Normalization uses stationary concentration supplied by final_conc'''

    idx = find_nearest(data, (start, end))

    x = data[:,0][idx[0]:idx[1]] - data[:,0][idx[0]]
    y = data[:,1][idx[0]:idx[1]]

    y_norm = y * (1./final_conc)

    return x, y_norm
    
def first_order_fitting(data, start, end, final_conc, plotting = False):
    '''Fitting of integrated, normalized first-order rate law to normalized baseline corrected data'''

    x, y = fitting_prep(data, start, end, final_conc)

    p_guess = np.ones(2)
    p = least_squares(fun=residual_generic, x0=p_guess, args=(x, y, first_order))
    p_solved = p.x

    y_fit = first_order(p_solved, x)

    if plotting == True:
        plt.plot(x, y, '.')
        plt.plot(x, y_fit, linewidth = 1)

    return p_solved

def first_order_fitting_without_normalization(p_guess, data, plotting = False):
    '''Fitting first_order_shift function to data, which has not been normalized'''

    x = data[:,0]
    y = data[:,1]

    p = least_squares(fun=residual_generic, x0=p_guess, args=(x, y, first_order_shift))
    y_fit = first_order_shift(p.x, x)

    if plotting == True:
        plt.plot(x, y, '.')
        plt.plot(x, y_fit, linewidth = 1)

    return p.x

def baseline_feature_function(p, x):
    '''Combined function describing baseline and feature'''

    return poly(p[:-3], x) + first_order_shift(p[-3:], x)

def combined_feature_fitting(p_guess, data, return_full = False, plotting = False):
    '''Fitting combined baseline_feature function to data. Lenght of p_guess determines order of polynomial
    used for baseline function (order polynomial = len(p_guess) - 4)'''

    x = data[:,0]
    y = data[:,1]

    p = least_squares(fun=residual_generic, x0=p_guess, args=(x, y, baseline_feature_function), method = 'lm')

    residual = residual_generic(p.x, x, y, baseline_feature_function)
    res = np.sum(residual**2)

    if return_full == False:
        return res

    else:
        y_fit = baseline_feature_function(p.x, x)
        baseline = poly(p.x[:-3], x)

        if plotting == True:
            plt.plot(x, y, '.')
            plt.plot(x, y_fit)
            plt.plot(x, baseline)

        return x, y, y_fit, baseline, p.x

def alternating_combined_fitting(p_guess, data, order_poly, convergence_threshold = 1e-9, max_iter = 500, plotting = False):
    '''Combined fitting of baseline and feature by alternating between baseline and feature fitting.
    First, a polynomial is fitted to the feature subtracted data using ordinary least squares.
    Then, the feature function is fitted to the baseline subtracted data using non linear least squares.
    This process is iterated until the change in the residual sum of squares between the last two iteration is below the convergence threshold or
    the number of iterations has reached the max_iter limit.'''

    diff = 1.  # initialize difference between last two residual sum of squares
    count = 0   # initialize iteration count

    x = data[:,0]
    y = data[:,1]

    feature = np.zeros(len(x))
    residuals = [diff]

    while abs(diff) > convergence_threshold and count < max_iter:

        p_poly = polynomial_regression(x, y - feature, order_poly)
        baseline = poly(p_poly, x)

        p_feature = least_squares(fun=residual_generic, x0=p_guess, args=(x, y - baseline, first_order_shift), method = 'lm')
        feature = first_order_shift(p_feature.x, x)

        residual = residual_generic(np.r_[p_poly, p_feature.x], x, y, baseline_feature_function)
        residual = np.sum(residual**2)
        residuals.append(residual)
        
        diff = residuals[-2] - residuals[-1]
        count += 1

    if plotting == True:
        plt.plot(x, y, '.')
        plt.plot(x, baseline)
        plt.plot(x, feature + baseline)

    return p_poly, p_feature

def fitting_success(data, cut_off, samples):
    '''Evaluates if fitting attempts were successful by comparing them against the cut_off. Returns percentage of
    successful attempts.'''

    data = np.asarray(data)
    success = data[data < cut_off]
    return 100. * (float(len(success))/float(samples))

def sample_performance(data, order_poly, sigma, samples, cut_off_combined = 0.0003, cut_off_ele = 0.0002):
    '''Performs combined fitting and LBC for provided data using polynomial of order order_poly to describe 
    baseline. Initial guesses for combined fitting and LBC are generated using random samples from the absolute values of a 
    normal distribution with the sigma value 'sigma'. Samples controls the the number of fitting attempts that will be performed.
    For each attempt, the sum of square of residuals is evaluated. If it is below the cut off (cut_off_combined for combined fitting,
    cut_off_ele LBC), the fitting attempt is classified as succesful.

    Function returns percentage of sucessful fitting attempt for the provided sigma value'''

    res_arr_combined = []
    res_arr_elevator = [] 

    for _ in range(samples):
        p_guess_combined = abs(np.random.normal(0., sigma, order_poly + 4))
        p_guess_combined[-1] = 2073.
        p_guess_elevator = abs(np.random.normal(0., sigma, order_poly + 2))

        res_combined = combined_feature_fitting(p_guess_combined, data, return_full = False)
        res_elevator = elevator_function_fitting(data, 2000., 4500., 4, return_full = False, p_guess = p_guess_elevator)

        res_arr_combined.append(res_combined)
        res_arr_elevator.append(res_elevator)

    return fitting_success(res_arr_combined, cut_off_combined, samples), fitting_success(res_arr_elevator, cut_off_ele, samples)

def sigma_sampling(data, order_poly, sigma, samples, plotting = False):
    '''Looping over array of sigma values to perform sample_performance function. Returns array of success percantages for each
    sigma value'''

    combined_arr = []
    ele_arr = []

    for i in sigma:
        comb, ele = sample_performance(data, order_poly, i, samples)

        combined_arr.append(comb)
        ele_arr.append(ele)

    combined_arr = np.asarray(combined_arr)
    ele_arr = np.asarray(ele_arr)

    if plotting == True:
        fig, ax = plt.subplots()
        width = 0.35

        x = np.arange(len(sigma))

        rects_1 = ax.bar(x - width/2, ele_arr, width, label = 'LBC')
        rects_2 = ax.bar(x + width/2, combined_arr, width, label = 'Combined')

        ax.set_ylabel('% Successful Fittings')
        ax.set_xticks(x)
        ax.set_xticklabels(sigma)
        ax.set_xlabel('$ \sigma $ of Initial Guess Distribution')

        autolabel(rects_1, ax)
        autolabel(rects_2, ax)

        legend_a = ax.legend(loc = 'center right')

    return combined_arr, ele_arr

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}%'.format(int(height)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize = 8)

def main():
    '''Imports raw H2O2 disproportionation data, shortens it to exclude outlier datapoints in first 500 s, performs LBC, pre- and post-signal fitting
    for baseline correction and analyzes all obtained baseline corrected signals by fitting integrated, normalized,
    first order rate law. All signals are normalized using stationary concentration obtained using LBC. 

    Analysis is also performed on non-normalized data obtained using LBC, pre-feature fitting and post-feature fitting by fitting
    the first_order_shift function.

    Combined fitting is performed using the sum of polynomial baseline and the first_order_shift function. Due to the
    necessity for a reasonable initial guess for this approach, the guess for the polynomial parameters is taken from the obtimized LBC
    solution.

    Guess dependence of combined fitting and LBC is evaluated by the sigma_sampling call, which plots the percantage of successful fittings
    for different initial guesses (initial guesses are random sanmples for the absolute value of a normal distribution with sigma value sigma)

    Use plotting flags to see different analysis stages/operations. These results are shown in Figure 3 (manuscript) and SI Figure 4.'''

    data = import_txt('170529_JS_378.txt', 1, 2)
    idx = find_nearest(data, 500.)[0]
    data_short = data[idx:]

    data_ele, baseline_ele, final_conc, p_ele_full = elevator_function_fitting(data_short, 2000., 4500., 4, plotting = False)
    data_pre = pre_signal_fitting(data_short, 2000., 1, plotting = False)
    data_post = post_signal_fitting(data_short, 4500., 2, plotting = False)

    p_ele = first_order_fitting(data_ele, 2089., 3000., final_conc, plotting = False)
    p_pre = first_order_fitting(data_pre, 2089., 3000., final_conc, plotting = False)
    p_post = first_order_fitting(data_post, 2089., 3000., final_conc, plotting = False)

    p_guess = np.r_[np.random.rand(2), 2073.]  # initial guess for fitting non-normalized data, first two parameters are random guesses for [A0] and k, third parameter is guess for feature start at t = 2073.

    p_ele_nn = first_order_fitting_without_normalization(p_guess, data_ele, plotting = False)
    p_pre_nn = first_order_fitting_without_normalization(p_guess, data_pre, plotting = False)
    p_post_nn = first_order_fitting_without_normalization(p_guess, data_post, plotting = False)

    p_guess_combined = np.r_[p_ele_full, np.random.rand(1), 2073.] # generating initial guess for combined fitting by using polynomial parameters and feature magnitude obtained using LBC, adding a random value for k guess and addig 2073 as guess for feature start time
    #p_guess_combined = np.r_[np.random.rand(3), final_conc, np.random.rand(1), 2073.]

    p_combined = combined_feature_fitting(p_guess_combined, data_short, return_full = True, plotting = False)
    alternating_combined_fitting(np.array([1., 1., 2073.]), data_short, 10, plotting = True)

    print('Results using normalized data, stationary O2 concentration:', final_conc)
    print('LBC: k = ', p_ele[1])
    print('Pre-Feature Fitting: k = ', p_pre[1])
    print('Post-Feature Fitting: k = ', p_post[1])

    print('Results using non-normalized data:')
    print('LBC: [A0] =', p_ele_nn[0], 'k =', p_ele_nn[1])
    print('Pre-Feature Fitting: [A0] =', p_pre_nn[0], 'k =', p_pre_nn[1])
    print('Post-Feature Fitting: [A0] =', p_post_nn[0], 'k =', p_post_nn[1])
    print('Combined Fitting: [A0] =', p_combined[-1][-3], 'k =', p_combined[-1][-2])

    sigma = np.array([0.001, 0.01, 0.1, 1., 10., 100.])
    comb, ele = sigma_sampling(data_short, 4, sigma, 50, plotting = True)

    plt.show()

if __name__ == '__main__':
    main()