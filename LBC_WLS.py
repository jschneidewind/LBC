import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import LBC as lbc

def logistic(m, spn, x):
	return (1./(1 + np.exp(spn*(x - m))))

def LBC_WLS(x, y, m, spn, order_poly, w):
	'''Weighted Least Square Implementation of LBC with polynomial baseline. w is a vector of the same length as x specifying the weights for
	each residual'''

	X_raw = np.tile(x, (order_poly+1, 1))
	powers = np.arange(0, order_poly+1)
	X = np.power(X_raw.T, powers)

	logit_values = logistic(m, spn, x)
	X = np.c_[X, logit_values]

	W = np.diag(w)

	a = inv(np.dot(np.dot(X.T, W), X))
	a = np.dot(np.dot(a, X.T), W)
	coef = np.dot(a, y)

	return coef

def LBC_WLS_fitting(data, start, end, order_poly, weight = 1., plotting = False):
	'''LBC fitting function using weighted least square implementation. Specifying weight to be a vector of the same length
	as x enables weighted least square, otherwise ordinary least square is performed. Order_Poly specifies the order
	of the polynomial used to describe the baseline.'''

	idx = lbc.find_nearest(data, (start, end))

	baseline = np.r_[data[:idx[0]], data[idx[1]:]]
	x = baseline[:,0]
	y = baseline[:,1]

	m = (data[:,0][idx[0]] + data[:,0][idx[1]]) / 2  # sigmoidal midpoint midway between pre- and post-signal intervals
	x_75 = (data[:,0][idx[1]] - m) / 2
	spn = np.log(1./9e6) / x_75    # logistic growth rate set so that curvature of logistic function does not interfere with baseline fitting

	if weight == 1.:
		weight = np.ones(len(x)) * weight

	p = LBC_WLS(x, y, m, spn, order_poly, weight)

	baseline = lbc.poly(p[:-1], data[:,0])
	lbc_fit = baseline + lbc.logistic(p[-1], m, spn, data[:,0])
	y_corrected = data[:,1] - baseline

	if plotting == True:
		plt.plot(data[:,0], data[:,1], '.', markersize = 2)
		plt.plot(data[:,0], lbc_fit)	
		plt.plot(data[:,0], baseline)
		plt.show()
		print(p)

	return baseline, lbc_fit, y_corrected, p

def main():
	np.random.seed(1)
	data = lbc.synthetic_data(lower_l = -2., upper_l = 4., no_dp = 1000, c0 = 1., k = 4., a = [0.4, 0.05, -0.007], sigma = 0.02)
	LBC_WLS_fitting(data, 0., 2., 2, plotting = True)

if __name__ == '__main__':
	main()#