import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

xTest = pd.read_csv('hw3xTest.csv',header=None)
xTrain = pd.read_csv('hw3xTrain.csv',header=None)
xValidation = pd.read_csv('hw3xValidation.csv',header=None)

yTest = pd.read_csv('hw3yTest.csv',header=None)
yTrain = pd.read_csv('hw3yTrain.csv',header=None)
yValidation = pd.read_csv('hw3yValidation.csv',header=None)


xTest = xTest.drop(xTest.columns[0], axis=1)
xTrain = xTrain.drop(xTrain.columns[0], axis=1)
xValidation = xValidation.drop(xValidation.columns[0], axis=1)

def  lambda_0(xTest,xTrain,xValidation,yTest,yTrain,yValidation):
    omega_true=np.array([2, 1,0,0,0,-0.5,0,0,2,0,3,0,0,0,0,0,0])
    
    M = len(omega_true)
    
    lambdas=np.array([0])
    
    xxtop = (xTrain.T).dot(xTrain)
    xtopy = (xTrain.T).dot(yTrain)
    
    omega_o = np.linalg.solve(xxtop, xtopy)  # omega_o = (x'x)^-1 * x'y
    #omega_o = np.linalg.inv(xxtop.T.dot(xxtop)).dot(xxtop.T).dot(xtopy)
    
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(omega_o, "b-",label =  'Estimated Weights')
    ax.plot(omega_true, "r-",label='True Weights')
    plt.xlabel(r'$Dimension$')
    plt.ylabel(r'$Weights$')
    plt.title(r'Estimated weights for lambda = 0 ')
    ax.legend(loc='upper center',bbox_to_anchor=(0.5, -0.17),shadow=True, ncol=2)
    plt.show()
    return

def  many_lambdas(xTest,xTrain,xValidation,yTest,yTrain,yValidation):    
    lambdas=[1,5,10,25,50,75,100,250,500,750,1000]
    
    xxtop = (xTrain.T).dot(xTrain)
    xtopy = (xTrain.T).dot(yTrain)
    
    omega = []   
    errorTrain = []   
    errorTest = []   
    errorTest = []   
    errorValidation = []   
    
    for k in range( len(lambdas)):
        omega.append( np.linalg.solve((xxtop + lambdas[k] * np.eye( len(xxtop))) , xtopy) )     #omega(k) = (x'x + λI)^-1 *x'y
        ypredTrain     = (omega[k].T.dot(xTrain.T)).T                                            #ypredTrain = (w * x')'
        ypredTest      = (omega[k].T.dot(xTest.T )).T                                            #ypredTest  = (w*x')'
        ypredValidation= (omega[k].T.dot(xValidation.T )).T                                      #ypredValidation = = (w*x')'
        errorTrain.append(   np.mean( (ypredTrain -yTrain )**2 ).T )
        errorTest.append(    np.mean((ypredTest -yTest )**2 ).T )
        errorValidation.append( np.mean((ypredValidation-yValidation)**2).T )
        
    fig = plt.figure()
    ax = plt.subplot(111)    
    ax.set_xscale('log')
    ax.plot(lambdas, errorTrain, "b-",label ='Train')
    ax.plot(lambdas, errorTest,  "r-",label ='Test')
    ax.plot(lambdas, errorValidation,  "g-",label ='Validation')
    plt.xlabel(r'$Lambda$')
    plt.ylabel(r'$MSE$')
    plt.title(r'MSE for various values of lambda')
    ax.legend(loc='upper center',bbox_to_anchor=(0.5, -0.17),shadow=True, ncol=3)
    plt.show()
    
    bestMSE = np.min(errorTest)
    bestlambdaindex = int(  np.where(errorTest == bestMSE)[0] )
    best_lambda=lambdas[bestlambdaindex];
    best_omega=omega[bestlambdaindex];
    
    bestValid = (omega[bestlambdaindex].T.dot(xValidation.T )).T  
    
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(yValidation, "b-",label =  'True yValidation')
    ax.plot(bestValid, "r-",label='Best prediction')
    #plt.axis([170,200,-30, 30])
    plt.xlabel(r'$Elements$')
    plt.ylabel(r'$Value$')
    plt.title(r'Estimated weights for lambda = 0 ')
    ax.legend(loc='upper center',bbox_to_anchor=(0.5, -0.17),shadow=True, ncol=2)
    plt.show()
    return best_omega

lambda_0(xTest,xTrain,xValidation,yTest,yTrain,yValidation)
many_lambdas(xTest,xTrain,xValidation,yTest,yTrain,yValidation)    

xTest = pd.read_csv('hw3xTest.csv',header=None)
xTrain = pd.read_csv('hw3xTrain.csv',header=None)
xValidation = pd.read_csv('hw3xValidation.csv',header=None)

lambda_0(xTest,xTrain,xValidation,yTest,yTrain,yValidation)
best_omega=many_lambdas(xTest,xTrain,xValidation,yTest,yTrain,yValidation) 

omega_true=np.array([2, 1,0,0,0,-0.5,0,0,2,0,3,0,0,0,0,0,0])

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(best_omega, "b-",label =  'Estimated Weights')
ax.plot(omega_true, "r-",label='True Weights')
plt.xlabel(r'$Dimension$')
plt.ylabel(r'$Weights$')
plt.title(r'Estimated weights for best lambda')
ax.legend(loc='upper center',bbox_to_anchor=(0.5, -0.17),shadow=True, ncol=2)
plt.show()