
# Importing various packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

x = np.random.rand(100,1)
y = 5*x*x + 0.1*np.random.randn(100,1)


xb = np.c_[np.ones((100,1)), x]
beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y)
xnew = np.array([[0],[2]])
xbnew = np.c_[np.ones((2,1)), xnew]
ypredict = xbnew.dot(beta)

plt.plot(xnew, ypredict, "r-")
plt.plot(x, y ,'ro')
plt.axis([0,1.0,-1, 6.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Simple Linear Regression')
plt.show()


linreg = LinearRegression()
linreg.fit(x,y)
xnew = np.array([[0],[2]])
ypredict = linreg.predict(xnew)

plt.plot(xnew, ypredict, "r-")
plt.plot(x, y ,'ro')
plt.axis([0,1.0,-1, 6.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Scikit-learn linear regression ')
plt.show()

ypredict = linreg.predict(x)

print('The intercept alpha: ', linreg.intercept_)
print('Coefficient beta : ', linreg.coef_)

# The mean squared error                               
print("Mean squared error: %.2f" % mean_squared_error(y, ypredict))

# Explained variance score: 1 is perfect prediction                                 
print('Variance score: %.2f' % r2_score(y, ypredict))