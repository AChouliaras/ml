# imports
import pandas as pd

def predict_sales(radio, weight, bias):
    return weight*radio + bias

def cost_function(radio, sales, weight, bias):
    companies = len(radio)
    total_error = 0.0
    
    for i in range(companies):
        total_error += (sales[i] - (weight*radio[i] + bias))**2
    return total_error / companies

def update_weights(radio, sales, weight, bias, learning_rate):
    weight_deriv = 0
    bias_deriv = 0
    companies = len(radio)

    for i in range(companies):
        # Calculate partial derivatives
        # -2x(y - (mx + b))
        weight_deriv += -2*radio[i] * (sales[i] - (weight*radio[i] + bias))

        # -2(y - (mx + b))
        bias_deriv += -2*(sales[i] - (weight*radio[i] + bias))

    # We subtract because the derivatives point in direction of steepest ascent
    weight -= (weight_deriv / companies) * learning_rate
    bias -= (bias_deriv / companies) * learning_rate

    return weight, bias

def train(radio, sales, weight, bias, learning_rate, iters):
    cost_history = []

    for i in range(iters):
        weight,bias = update_weights(radio, sales, weight, bias, learning_rate)

        #Calculate cost for auditing purposes
        cost = cost_function(radio, sales, weight, bias)
        cost_history.append(cost)
        
        # Log Progress
        if (i % 10 == 0 or i==1):
            print ("iter: %-2i" %i + " weight: %.2f" %weight + " bias: %.4f" %bias + " cost: %.2f" %cost)
    return weight, bias, cost_history

df = pd.read_csv('Advertising.csv')
print(df.shape)

companies = df['company'].values
radio = df['radio'].values
sales = df['sales'].values
weight = 0.0
bias   = 0.0001

learning_rate=0.000048
iters = 41

train(radio, sales, weight, bias, learning_rate, iters)
