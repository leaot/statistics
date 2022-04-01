### Procedure ads nonlinear regression diagnostics
### T.P. Leao April, 01, 2022 
### Share knowledge, do not charge for it


# dependencies
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from scipy.stats import t

# import external file 
ft =  pd.read_csv(r"~/code/python/faithful.txt", delim_whitespace = True)

# uncomment here to print dataset to terminal
#print(ft)

# define initial values for the parameters
# change values if the procedure does not converge
A = 100
gamma = 2
tau = 1

# vector of initial parameter values
InitPar = [A, gamma, tau]

# function to be fit - change for a custom function
def func(x, A, gamma, tau):
    return A/(1 + np.exp(-gamma*(x-tau)))

# fits data

popt, pcov = curve_fit(func, ft['eruptions'], ft['waiting'], p0 = InitPar, method = 'lm') 


print("--------------------------------------------------------------")
print("-------NONLINEAR REGRESSION DIAGNOSTICS-----------------------") 
print("--------------------------------------------------------------")
print("-------COVARIANCE MATRIX OF PARAMETERS------------------------")
print(pcov)
print("--------------------------------------------------------------\n \n")


print("---------FITTING PARAMETERS-----------------------------------")
print(popt)
print("-------------------------------------------------------------- \n \n ")

# degrees of freedom n - number of parameters
n = len(ft['eruptions'])
d = len(popt)
df = n-d

# standard error
se0 = np.sqrt(pcov[0,0])
se1 = np.sqrt(pcov[1,1])
se2 = np.sqrt(pcov[2,2])

# t value
t0 = popt[0]/np.sqrt(pcov[0,0])
t1 = popt[1]/np.sqrt(pcov[1,1])
t2 = popt[2]/np.sqrt(pcov[2,2])

# significance
# uses the sf function, equivalent (but more precise) than 1 - cdf
# see scipy.stats.t documentation
p0 = 2*(t.sf(np.abs(t0), df, loc=0, scale =1))
p1 = 2*(t.sf(np.abs(t1), df, loc=0, scale =1))
p2 = 2*(t.sf(np.abs(t2), df, loc=0, scale =1))

# macro for significance
def sigfun(x):
    if x > 0.05:
        return "ns"
    elif (x <= 0.05 and x > 0.01):
        return "*"
    elif (x <= 0.01 and x > 0.001):
        return "**"
    else:
        return "***" 

# prints statistics to terminal
print("--------SIGNIFICANCE OF THE PARAMETERS-------------------------") 
print("Parameter -- Estimate -- Std. Err. -- t -- pr>(|t|) -- Signif.") 
print("A           ", round(popt[0],4), "   ", round(se0,4)," ",round(t0,4), " ", np.format_float_scientific(p0, exp_digits=2, precision=4)," ", sigfun(p0)) 
print("gamma        ", round(popt[1],4), "   ",  round(se1,4),"  ", round(t1,4), " ", np.format_float_scientific(p1, exp_digits=2, precision=4)," ",  sigfun(p1))
print("tau          ", round(popt[2],4), "   ", round(se2,4)," ", round(t2,4), " ", np.format_float_scientific(p2, exp_digits=2, precision=4), " ", sigfun(p2))
print("----------------------------------------------------------------")
print("ns: > 0.05")
print("*:   0.05")
print("**:  0.01")
print("***: 0.001")
print("---------------------------------------------------------------- \n \n ")


# calculates the pseudo r2 from the correlation coefficient
r = np.corrcoef(ft['waiting'], func(ft['eruptions'], *popt))
r2 = r[0,1]**2

# prints R2 to terminal
print("--------PSEUDO R2-----------------------------------------------")
print("The R2 of predicted vs estimated values is: ", round(r2,4))
print("Use the R2 with caution in nonlinear regression")
print("----------------------------------------------------------------- \n \n ")

# calculates sum of squares of deviations SSQ and prints to terminal
SSQ = np.sqrt(np.sum((ft['waiting']- func(ft['eruptions'], *popt))**2)/df)
print("Sum of squares of the residuals: ", round(SSQ, 4))
print("Degrees of freedom: ", df)
print("----------------------------------------------------------------\n \n ")
print("---------------END OF PROBLEM-----------------------------------")



# simulated range for predicted plot
Sim = np.arange(np.min(ft['eruptions']), np.max(ft['eruptions']), np.abs(np.max(ft['eruptions']) - np.min(ft['eruptions']))/10000)

# plot - define paramters according to user need

plt.plot(ft['eruptions'], ft['waiting'], "*")
plt.plot(Sim, func(Sim, *popt), "--")
plt.xlabel("Duration of eruption")
plt.ylabel("Waiting time")
plt.show()

 
