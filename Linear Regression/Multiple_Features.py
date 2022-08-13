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
def gradient(w1,w2,b,x1,x2,y):
    
    dj_dw1=0.0
    dj_dw2=0.0
    dj_db=0.0
    m=len(x1)
    for i in range(m):
        dj_dw1 += ((w1*x1[i]+w2*x2[i]*b-y[i])*x1[i])
        dj_dw2 += ((w1*x1[i]+w2*x2[i]*b-y[i])*x2[i])
        dj_db += (w1*x1[i]+w2*x2[i]*b-y[i])

    dj_dw1=dj_dw1/(m)
    dj_dw2=dj_dw2/(m)
    dj_db=dj_db/(m)

    return  dj_db,dj_dw1,dj_dw2



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