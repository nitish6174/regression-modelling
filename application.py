import sys
from base import *

data = np.loadtxt('data/data1.txt', delimiter=',')

model = model_2var(data,10000,0.02)
print( "Initial cost taking theta as zero vector : " + str(model.compute_cost()) )
model.gradient_descent()
print( "Cost after running gradient descent for 100 steps is : " + str(model.compute_cost()) )
print( "Theta obtained from running gradient descent :\n" + str(model.get_theta()) )

# model.plot_points()
# model.plot_line()
model.plot_points_with_line()