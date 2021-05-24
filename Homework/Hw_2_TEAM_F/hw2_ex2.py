# imports
import pandas as pd
import numpy as np

def normalize(features):
    '''
    features     -   (200, 4)
    features.T   -   (4, 200)

    We transpose the input matrix, swapping
    cols and rows to make vector math easier
    '''

    for feature in features.T:
        fmean = np.mean(feature)
        frange = np.amax(feature) - np.amin(feature)

        #Vector Subtraction
        feature -= fmean

        #Vector Division
        feature /= frange

    return features

def predict(features, weights):
  '''
  features - (200, 4)
  weights - (4, 1)
  predictions - (200,1)
  '''
  return np.dot(features,weights)

def cost_function(features, targets, weights):
    
    #Features:(200,4)
    #Targets: (200,1)
    #Weights:(4,1)
    #Returns 1D matrix of predictions
    
    N = len(targets)

    predictions = predict(features, weights)

    # Matrix math lets use do this without looping
    sq_error = (predictions - targets)**2

    # Return average squared error among predictions
    return 1.0/(2*N) * sq_error.sum()

def update_weights_vectorized(X, targets, weights, lr):
    '''
    gradient = X.T * (predictions - targets) / N
    X: (200, 4)
    Targets: (200, 1)
    Weights: (4, 1)
    '''
    companies = len(X)

    #1 - Get Predictions
    predictions = predict(X, weights)
    #2 - Calculate error/loss
    error = targets - predictions
    #3 Transpose features from (200, 3) to (3, 200)
    # So we can multiply w the (200,1)  error matrix.
    # Returns a (3,1) matrix holding 3 partial derivatives --
    # one for each feature -- representing the aggregate
    # slope of the cost function across all observations
    gradient = np.dot(-X.T,  error)

    #4 Take the average error derivative for each feature
    gradient /= companies

    #5 - Multiply the gradient by our learning rate
    gradient *= lr
    
    #6 - Subtract from our weights to minimize cost
    weights -= gradient

    return weights

def train(features, targets, weights, lr, iters):
    cost_history = []

    for i in range(iters+1):
        weights = update_weights_vectorized(features, targets, weights, lr)

        #Calculate cost for auditing purposes
        cost = cost_function(features, targets, weights)
        cost_history.append(cost)
        
        # Log Progress
        if (i % 1000 == 0 or i==1):
            print ("iter: %-5i"         %i + 
                   " weights: TV: %.2f" %weights[1] + 
                   " radio: %.2f"       %weights[2] + 
                   " newspaper: %.2f"   %weights[3] + 
                   " bias: %.4f"        %weights[0] + 
                   " cost: %.2f"        %cost)
            
    return weights, bias, cost_history


df = pd.read_csv('Advertising.csv')
print(df.shape)

features = df[['TV', 'radio' , 'newspaper']]
features = normalize(features.values)
targets = df['sales'].values.reshape(-1, 1) 

B  = 0.0
W1 = 0.0
W2 = 0.0
W3 = 0.0
weights = np.array([
    [B ],   
    [W1],
    [W2],
    [W3]
])

bias = np.ones(shape=(len(features),1))
features = np.append(bias, features, axis=1)

lr = 0.0005
iters = 10000

train(features, targets, weights, lr, iters)