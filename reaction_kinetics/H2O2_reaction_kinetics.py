import numpy as np
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

def residual(p, m, spn, x, y):

    y_poly = poly(p[:-1], x)
    y_log = logistic(p[-1], m, spn, x)
    res = y + y_log - y_poly

    return res

def residual_generic(p, x, y, function):

    y_fit = function(p, x)
    res = y - y_fit

    return res

def elevator_function_fitting(data, start, end, order_poly, plotting = False):
    '''LBC function'''

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
    data_corr = np.c_[data[:,0], y_corrected]

    if plotting == True:
        plt.plot(x, y, '.', markersize = 2.0)
        plt.plot(data[:,0], y_fit, linewidth = 1.0)
        plt.plot(data[:,0], y_baseline, linewidth = 1.0)
        plt.plot(data[:,0], y_corrected)

    return data_corr[idx[0]:], y_baseline, -p_solved[-1]

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

    return np.c_[data[:,0], y_corrected][idx[0]:]

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

    return np.c_[data[:,0], y_corrected-np.mean(y_corrected[:10])]

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


def main():
    '''Imports raw H2O2 disproportionation data, shortens it to exclude outlier datapoints in first 500 s, performs LBC, pre- and post-signal fitting
    for baseline correction and analyzes all obtained baseline corrected signals by fitting integrated, normalized,
    first order rate law. All signals are normalized using stationary concentration obtained using LBC. Use plotting flags to see different
    analysis stages/operations. These results are shown in Figure 3 (manuscript) and SI Figure 4.'''

    data = import_txt('170529_JS_378.txt', 1, 2)
    idx = find_nearest(data, 500.)[0]
    data_short = data[idx:]

    data_ele, baseline_ele, final_conc = elevator_function_fitting(data_short, 2000., 4500., 4, plotting = False)
    data_pre = pre_signal_fitting(data_short, 2000., 1, plotting = False)
    data_post = post_signal_fitting(data_short, 4500., 2, plotting = False)

    first_order_fitting(data_ele, 2089., 3000., final_conc, plotting = True)
    first_order_fitting(data_pre, 2089., 3000., final_conc, plotting = False)
    first_order_fitting(data_post, 2089., 3000., final_conc, plotting = False)

    plt.show()

if __name__ == '__main__':
    main()