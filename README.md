# LBC

Logistic baseline correction (LBC) is a general background correction method for baseline-varying signals.

LBC.py contains the basic functions necessary for LBC along with the "elevator_function_fitting" function, which performs the actual LBC calculation. Input for this function is a 2D (NumPy) array, containg x values in column 0 and y values in column 1. Furthermore, signal start and end point have to be specified (in units of x, not as indices) along with the order of the polynomial used to describe the baseline.
Ouput of the function is a 2D NumPy array containing the original x values (in column 0) and the baseline corrected y values (in column 1).

In addition to the basic LBC module, the different subrepositories contain the scripts used to perform the synthetic data experiments, analysis of chemical reaction kinetics, qPCR analysis and X-ray absorption spectroscopy analysis.


# Dependencies

Basic LBC and reaction kinetics analysis require NumPy and SciPy, as well as Matplotlib for visualization.

qPCR analysis addtionally requires Pandas, statsmodels and timeit.

X-Ray absorption spectroscopy analysis requires Larch in addition to NumPy, SciPy and Matplotlib.


