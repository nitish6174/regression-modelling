# Regression modelling
###### Python project for automatically predicting values using Regression techniques

### About
The project aims to develop a module to take numerical/graphical data,
use linear regression techniques to find a linear relation among the datasets and use it to predict the values for future input.  

The current code simply reads data from a test file, and finds the best fitting (approximately) line for the data points.
The current code is only for 2 variable dataset.

### Using the project

1. Use Python 3 for running this project.
2. Modules used in code:
  * numpy (For array/matrix operations)
  * matplotlib (To plot graphs)
3. Clone/download the repository.
The datafile on which you want to run the code must follow syntax similar to the sample file present in data folder otherwise,
you need to make suitable changes in the ```application.py``` file.
4. Run ```python application.py``` in terminal to run the project.
  
### About the code
* ```base.py``` defines the linear regression model class which provides various functions to get data and for visualising.
* The class will soon be generalised to support dataset consisting of more variables. (Including a 3D plot for 3 variable dataset)
* ```application.py``` uses the ```base``` module to find a best fitting line for the data in a sample file.

## About the project author
#### Nitish Garg
B.Tech undergraduate (Computer Science & Engineering)  
IIT Guwahati  
India  

nitish.garg.6174@gmail.com  
www.linkedin.com/in/nitish6174
