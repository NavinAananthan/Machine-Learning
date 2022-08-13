import numpy as np
import pandas as pd
import seaborn as sns
from numpy import *
import matplotlib.pyplot as plt



# This function is based on statistics by appying appropriate formula and calculating it through mean
def stat(x,y):
    xmean=mean(x)
    ymean=mean(y)

    # calculating cross-deviation and deviation about x
    SS_XY=np.sum(y*x)-len(x)*xmean*ymean
    SS_XX=np.sum(x*x)-len(y)*xmean*xmean

    # calculating regression coefficients
    b_1 = SS_XY / SS_XX         #slope
    b_0 = ymean - b_1*xmean     #intercept

    return b_1,b_0



# This function is used to find the cost of the given predicted data and actual data the cost should be minimized
def cost(w,b,x,y):

    total_cost=0
    n=len(x)
    for i in range(n):
        total_cost += y[i]-(x[i]*w+b)

    return total_cost/(2*n)



# This function is used to compute the gradient of the given model or the funtion 
def gradient(w,b,x,y):
    
    dj_dw=0.0
    dj_db=0.0
    for i in range(len(x)):
        dj_dw += (w*x[i]+b-y[i])*x[i]
        dj_db += w*x[i]+b-y[i]

    dj_dw=dj_dw/(len(x))
    dj_db=dj_db/(len(x))

    return  dj_db,dj_dw



# We pass old value of the intercepts and get new value and iterate untill we get a good model for our linear regression model
def gradient_descent(w,b,x,y,iterations):

    for i in range(iterations):
        dj_db,dj_dw=gradient(w,b,x,y)
        w=w-0.01*(dj_dw)
        b=b-0.01*(dj_db)
        #cost=cost(w,b,x,y)
        print("Iteration:{} w:{} b:{}".format(i,w,b))

    return w,b



#This is an example data
car=pd.read_csv('E:\Python\Machine Learning\Linear Regression\carprices.csv')

corr=car.corr()
sns.heatmap(corr,annot=True)


# By heat map in seaborn library we use the mileage to determine the age of the car
x=car.Age
y=car.Mileage

# Initializing the intercepts to zero
w=0
b=0

# calculating the value for the intercepts 
w,b=gradient_descent(w,b,x,y,10)
#w1,b1=stat(x,y)

# Prediciting the y values
ypred=x*w+b

# Calculating the cost for the given data and predicted data
cost=cost(w,b,x,y)
print(cost)

# Plotting the data sets
plt.scatter(x,y)
plt.plot(x,ypred)
plt.show()