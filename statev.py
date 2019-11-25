import plot
import logistic_regression as lr

data = lr.load_data('iris_data.csv') # Load the data
plot.scatter_plot(data,['Iris-setosa','Iris-versicolor']) # Scatter plot of the data

X,y = lr.split(data) # Split into data and labels

X_train, X_test, y_train, y_test = lr.train_test_split(X,y) # Split all the data into training and testing set

theta = lr.SGD(X_train, y_train) # Run SGD to calculate optimal theta
print('\nCalculated theta:\n {}'.format(theta))

hypothesis = lr.predict(X_test,theta) # Test the model

lr.accuracy(hypothesis, y_test)

plot.boundary(data, ['Iris-setosa', 'Iris-versicolor'], theta) # Plot the decision boundary