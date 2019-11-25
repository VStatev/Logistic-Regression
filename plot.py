import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Scatter plot of the dataset
def scatter_plot(data, labels):
    X_zero = data[:50, :3]
    X_one = data[50:, :3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_zero[:,0], X_zero[:,1], X_zero[:,2], marker='o', color='b', label = labels[0])
    ax.scatter(X_one[:,0], X_one[:,1], X_one[:,2], marker='^', color='r', label = labels[1])

    ax.set_xlabel('Sepal Length')
    ax.set_ylabel('Sepal Width')
    ax.set_zlabel('Petal Length')
    ax.title.set_text('Iris features')
    ax.legend()
    plt.show()

# Scatter plot of the dataset with the separating hyperplane
def boundary(data, labels, theta):
    X_zero = data[:50, :3]
    X_one = data[50:, :3]

    x_plot = np.linspace(0,7,7)  # Points for surface
    y_plot = np.linspace(0,5,7)  # Points for surface

    xx,yy = np.meshgrid(x_plot, y_plot)  # x and y coordinates for the separating hyperplane

    zz = - (theta[0] + (theta[1]*xx) + (theta[2]*yy))/theta[3]  # decision boundary of the model

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_zero[:,0], X_zero[:,1], X_zero[:,2], marker='o', color='b', label = 'Iris-setosa')
    ax.scatter(X_one[:,0], X_one[:,1], X_one[:,2], marker='^', color='r', label = 'Iris-versicolor')
    ax.plot_surface(xx, yy, zz, cmap='winter', edgecolor='none')
    ax.set_xlabel('Petal Length')
    ax.set_ylabel('Petal Width')
    ax.set_zlabel('Sepal Length')
    ax.title.set_text('Iris features')
    ax.legend()
    plt.show()