import numpy as np 
import matplotlib.pyplot as plt

# Function to load dataset
def load_data(path):
    data = np.loadtxt(path,delimiter=',', converters={3: lambda s: 0 if(s.decode('utf-8') == 'Iris-setosa') else 1})
    return data

# Split Data into X and y 
def split(data):
    X = data[:,:3] 
    X = np.insert(X, 0, 1, axis=1)  # Inserting bias column

    y = data[:, 3]
    y = y.reshape(-1,1)  # Change y from row to column vector
    return X,y


# Split data into training and testing set to check accuracy of model
def train_test_split(X, y):
    random = np.random.choice(X.shape[0], 20, replace=False)
    X_test = X[random, :]
    y_test = y[random, :]

    X_train = np.delete(X, random, axis=0)
    y_train = np.delete(y, random, axis=0)

    return X_train, X_test, y_train, y_test


# Define Sigmoid Function for calculating probability of class 1
def sigmoid(X, theta):  
    return 1/(1 + np.exp(-X@theta))

# Calculate the loss function of the model 
def calculate_loss(X, y, theta):  
    hypothesis = sigmoid(X, theta)
    return (-y.T @ np.log(hypothesis) - (1-y).T @ np.log(1-hypothesis))

# Calculating the gradient used in SGD 
def calculate_gradient(X, y, theta): 
    hypothesis = sigmoid(X, theta)
    return X.T@(hypothesis-y)

# Implementation of SGD
def SGD(X, y, T=100, k=20, eta=0.1):  
    theta = np.zeros((X.shape[1],1))
    loss = np.zeros((T,1)) 
    eta_t = eta
    for t in range(1,T):
        random = np.random.choice(X.shape[0], k, replace=False)  # Randomly take k numbers from the number of rows in X 
        X_random = X[random, :]
        y_random = y[random]
        loss[t] = calculate_loss(X_random,y_random,theta)  # Save the loss function for every theta, debug purpose
        theta = theta - eta_t * calculate_gradient(X_random, y_random, theta)  # Estimating new theta
        eta_t = eta/np.sqrt(t)   # Update learning rate
    plt.plot(loss)   # Plot of loss function history, debug purpose
    plt.xlabel('# of Iteration')
    plt.ylabel('Loss')
    plt.title('Loss of Model')
    plt.show()
    return theta

# Generating hypothesis 
def predict(X, theta):
    probabilities = sigmoid(X,theta)
    
    predictions = np.zeros_like(probabilities)
    # Convert probabilities into labels
    for i in range (0,probabilities.shape[0]):
        if probabilities[i] >= 0.5:
            predictions[i] = 1
        else:
            predictions[i] = 0
    return predictions

# Calculate accuracy of trained model
def accuracy(hypothesis, y):
    correct = 0

    # Compare hypothesis with y
    for i in range(0, y.shape[0]):
        if(hypothesis[i] == y[i]):
            correct = correct + 1

    accuracy = correct / y.shape[0]

    print('Accuracy of model = {}'.format(accuracy))



if __name__ == "__main__":
    data = load_data('iris_data.csv')
    X, y = split(data)
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    theta = SGD(X_train, y_train)
    accuracy(predict(X_test, theta), y_test)
