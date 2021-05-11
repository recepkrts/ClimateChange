import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns; sns.set()


yearsBase, meanBase = np.loadtxt('5-year-mean-1951-1980.csv',
                                 delimiter=',', usecols=(0, 1),
                                 unpack=True)
years, mean = np.loadtxt('5-year-mean-1882-2014.csv',
                         delimiter=',',
                         usecols=(0, 1),
                         unpack=True)
#######################################################################################################
    # Creates a linear regression from the data points ,b =
    #np.polyfit(yearsBase, meanBase, 1)
    #
    # This is a simple y = mx + b line function
    #ef f(x):
    #   return m*x + b
    #
    # This generates the same scatter plot as before, but adds a line plot using the function above
    #lt.scatter(yearsBase, meanBase)
    #lt.plot(yearsBase, f(yearsBase))
    #lt.title('scatter plot of mean temp difference vs year')
    #lt.xlabel('years', fontsize=12)
    #lt.ylabel('mean temp difference', fontsize=12)
    #lt.show()
    #
    # Prints text to the screen showing the computed values of m and b
    #rint(' y = {0} * x + {1}'.format(m, b))
    #lt.show()
    # Pick the Linear Regression model and instantiate it odel =
    #LinearRegression(fit_intercept=True)
###########################################################################################################
    # Fit/build the model
    #odel.fit(yearsBase[:, np.newaxis], meanBase)
    #ean_predicted = model.predict(yearsBase[:, np.newaxis])
    #
    # Generate a plot like the one in the previous exercise
    #lt.scatter(yearsBase, meanBase)
    #lt.plot(yearsBase, mean_predicted)
    #lt.title('scatter plot of mean temp difference vs year')
    #lt.xlabel('years', fontsize=12)
    #lt.ylabel('mean temp difference', fontsize=12)
    #lt.show()
    #
    #rint(' y = {0} * x + {1}'.format(model.coef_[0], model.intercept_))
############################################################################################################
plt.scatter(years, mean)
plt.title('scatter plot of mean temp difference vs year')
plt.xlabel('years', fontsize=12)
plt.ylabel('mean temp difference', fontsize=12)
sns.regplot(yearsBase, meanBase)
plt.show()
