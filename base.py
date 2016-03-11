import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d

def get_2d_scatterplot(data_2var,title="2 variable plot",xlabel="x-axis",ylabel="y-axis"):
	if data_2var.shape[1]==2:
		plt.scatter(data_2var[:, 0], data_2var[:, 1], marker='o', c='b')
		plt.title(title)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		return plt
	else:
		print("This function is for 2 variable data matrix only")
		return None

class model_2var(object):

	def __init__(self, data, iterations=100, alpha=0.01):

		self.data = data
		X_temp = data[:, 0]
		self.y = data[:, 1]
		self.n = self.y.size
		self.X = np.ones(shape=(self.n, 2))
		self.X[:, 1] = X_temp

		self.theta = np.zeros(shape=(2, 1))

		self.iterations = iterations
		self.alpha = alpha

	def compute_cost(self):
		predictions = (self.X).dot(self.theta).flatten()
		sqErrors = ((predictions - self.y) ** 2)
		cost = (1.0 / (2 * self.n)) * sqErrors.sum()
		return cost

	def gradient_descent(self,num_iterations=None):

		if num_iterations is None:
			num_iterations = self.iterations

		self.cost_history = np.zeros(shape=(num_iterations, 1))

		for i in range(num_iterations):

			predictions = (self.X).dot(self.theta).flatten()
			errors_x1 = (predictions - self.y) * self.X[:, 0]
			errors_x2 = (predictions - self.y) * self.X[:, 1]

			self.theta[0][0] = self.theta[0][0] - ( self.alpha*(1.0/self.n) * errors_x1.sum() )
			self.theta[1][0] = self.theta[1][0] - ( self.alpha*(1.0/self.n) * errors_x2.sum() )

			self.cost_history[i,0] = self.compute_cost()

	def get_theta(self):
		return self.theta

	def get_predicted_value(self,x):
		x_matrix = np.ones(shape=(x.size,2))
		x_matrix[:,1] = x
		return x_matrix.dot(self.theta).flatten()


	def get_line(self):
		min_x = np.min(self.data[:, 1])
		max_x = np.max(self.data[:, 1])
		x_values = np.linspace(min_x,max_x)
		y_values = self.get_predicted_value(x_values)
		line, = plt.plot(x_values, y_values, '-', linewidth=1)
		dashes = [10, 5, 100, 5]  # 10 points on, 5 off, 100 on, 5 off
		line.set_dashes(dashes)
		return plt

	def plot_points(self):
		plt = get_2d_scatterplot(self.data)
		if plt==None:
			pass
		else:
			plt.show()

	def plot_line(self):
		min_x = np.min(self.data[:, 1])
		max_x = np.max(self.data[:, 1])
		x_values = np.linspace(min_x,max_x)
		y_values = self.get_predicted_value(x_values)
		line, = plt.plot(x_values, y_values, '-', linewidth=1)
		dashes = [10, 5, 100, 5]  # 10 points on, 5 off, 100 on, 5 off
		line.set_dashes(dashes)
		plt.show()

	def plot_points_with_line(self):
		plt = get_2d_scatterplot(self.data)
		if plt==None:
			pass
		else:
			min_x = np.min(self.data[:, 1])
			max_x = np.max(self.data[:, 1])
			x_values = np.linspace(min_x,max_x)
			y_values = self.get_predicted_value(x_values)
			line, = plt.plot(x_values, y_values, '-', linewidth=1)
			dashes = [10, 5, 100, 5]  # 10 points on, 5 off, 100 on, 5 off
			line.set_dashes(dashes)
			plt.show()